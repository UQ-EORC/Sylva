// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Interpolation between point clouds and rasters.
//!
//! * Attributes between clouds: for every target point, the index of the
//!   source point whose value it takes ([`nearest_indices`],
//!   [`majority_indices`]) or an inverse-distance weighted mean
//!   ([`idw_values`]). A k-d tree is built on the source and the targets are
//!   processed in parallel; every target is computed independently, so the
//!   results do not depend on the number of threads.
//! * Points to rasters ([`grid`]): inverse distance weighting, linear
//!   interpolation on a Delaunay triangulation (TIN) or natural-neighbour
//!   (Sibson) interpolation, evaluated at cell centres.
//! * Rasters onto points ([`sample_raster`]): bilinear or nearest-cell values,
//!   NaN outside the raster.

use rayon::prelude::*;
use spade::handles::FixedVertexHandle;
use spade::{DelaunayTriangulation, FloatTriangulation, HasPosition, HintGenerator, Point2, Triangulation};

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::raster::Raster;
use crate::util::spatial::KdTree;
use crate::Point;

/// Targets handled by one parallel task.
const BLOCK: usize = 16_384;

// ------------------------------------------------------------------ neighbours

/// k-d tree over the finite points of a source cloud, remembering their
/// original indices.
struct Source {
    tree: KdTree,
    index: Vec<usize>,
}

impl Source {
    fn new(points: &[Point]) -> Self {
        let index: Vec<usize> = (0..points.len()).filter(|&i| points[i].iter().all(|v| v.is_finite())).collect();
        let kept: Vec<Point> = index.iter().map(|&i| points[i]).collect();
        Source { tree: KdTree::new(&kept), index }
    }

    fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Up to `k` nearest source points within `max_distance`, as
    /// `(original index, distance)` sorted by distance. Empty for a
    /// non-finite query.
    fn knn(&self, p: &Point, k: usize, max_distance: Option<f64>) -> Vec<(usize, f64)> {
        if self.is_empty() || !p.iter().all(|v| v.is_finite()) {
            return Vec::new();
        }
        let mut nb = self.tree.knn(p, k.min(self.index.len()));
        if let Some(d) = max_distance {
            nb.retain(|&(_, dist)| dist <= d);
        }
        for e in &mut nb {
            e.0 = self.index[e.0];
        }
        nb
    }

    fn nearest(&self, p: &Point, max_distance: Option<f64>) -> Option<usize> {
        if self.is_empty() || !p.iter().all(|v| v.is_finite()) {
            return None;
        }
        let (i, d) = self.tree.nearest(p)?;
        match max_distance {
            Some(m) if d > m => None,
            _ => Some(self.index[i]),
        }
    }
}

fn check_k(k: usize, max_distance: Option<f64>) -> Result<()> {
    if k == 0 {
        return Err(Error::invalid("k must be at least 1"));
    }
    check_max_distance(max_distance)
}

fn check_max_distance(max_distance: Option<f64>) -> Result<()> {
    match max_distance {
        Some(d) if d.is_nan() || d < 0.0 => Err(Error::invalid(format!("max_distance must be a non-negative number, got {d}"))),
        _ => Ok(()),
    }
}

/// Fill `m` output columns of length `n` in parallel blocks of targets. `f`
/// gets the first target of the block and the block's slice of each column.
fn fill_columns<T: Send + Clone>(n: usize, m: usize, init: T, f: impl Fn(usize, &mut [&mut [T]]) + Sync) -> Vec<Vec<T>> {
    let mut outs: Vec<Vec<T>> = (0..m).map(|_| vec![init.clone(); n]).collect();
    let mut blocks: Vec<Vec<&mut [T]>> = (0..n.div_ceil(BLOCK)).map(|_| Vec::with_capacity(m)).collect();
    for col in outs.iter_mut() {
        for (b, chunk) in col.chunks_mut(BLOCK).enumerate() {
            blocks[b].push(chunk);
        }
    }
    blocks.into_par_iter().enumerate().for_each(|(b, mut cols)| f(b * BLOCK, &mut cols));
    outs
}

/// Index of the nearest source point for every target, or -1 where the
/// target is not finite, the source has no finite point or the nearest one
/// is farther than `max_distance`.
pub fn nearest_indices(source: &[Point], target: &[Point], max_distance: Option<f64>) -> Result<Vec<i64>> {
    check_max_distance(max_distance)?;
    let src = Source::new(source);
    Ok(target.par_iter().map(|p| src.nearest(p, max_distance).map_or(-1, |i| i as i64)).collect())
}

/// Keys that compare equal exactly when two attribute values are equal
/// (every NaN is one key, and -0.0 equals 0.0), for majority votes.
pub fn label_keys(a: &Attr) -> Vec<u64> {
    let float = |v: f64| -> u64 {
        if v.is_nan() {
            u64::MAX
        } else {
            (v + 0.0).to_bits()
        }
    };
    match a {
        Attr::F64(v) => v.iter().map(|&x| float(x)).collect(),
        Attr::F32(v) => v.iter().map(|&x| float(x as f64)).collect(),
        Attr::I64(v) => v.iter().map(|&x| x as u64).collect(),
        Attr::I32(v) => v.iter().map(|&x| x as i64 as u64).collect(),
        Attr::U32(v) => v.iter().map(|&x| x as u64).collect(),
        Attr::U16(v) => v.iter().map(|&x| x as u64).collect(),
        Attr::U8(v) => v.iter().map(|&x| x as u64).collect(),
        Attr::I8(v) => v.iter().map(|&x| x as i64 as u64).collect(),
        Attr::Bool(v) => v.iter().map(|&x| x as u64).collect(),
    }
}

/// For each label column and each target, the index of the source point that
/// carries the most common label among the target's `k` nearest source
/// points within `max_distance`, or -1 if there is none. Ties go to the label
/// of the nearest point, and the index returned is that of the nearest point
/// with the winning label.
pub fn majority_indices(source: &[Point], target: &[Point], labels: &[Vec<u64>], k: usize, max_distance: Option<f64>) -> Result<Vec<Vec<i64>>> {
    check_k(k, max_distance)?;
    for l in labels {
        if l.len() != source.len() {
            return Err(Error::invalid(format!("label column has {} values for {} source points", l.len(), source.len())));
        }
    }
    let src = Source::new(source);
    Ok(fill_columns(target.len(), labels.len(), -1i64, |start, cols| {
        let mut counts: Vec<usize> = Vec::with_capacity(k);
        for t in 0..cols.first().map_or(0, |c| c.len()) {
            let nb = src.knn(&target[start + t], k, max_distance);
            if nb.is_empty() {
                continue;
            }
            for (col, lab) in cols.iter_mut().zip(labels) {
                // counts[j] = votes for the label of neighbour j, counted at its
                // first (nearest) occurrence; later duplicates count zero.
                counts.clear();
                for (j, &(i, _)) in nb.iter().enumerate() {
                    let key = lab[i];
                    match nb[..j].iter().position(|&(q, _)| lab[q] == key) {
                        Some(first) => {
                            counts[first] += 1;
                            counts.push(0);
                        }
                        None => counts.push(1),
                    }
                }
                let mut best = 0;
                for j in 1..counts.len() {
                    if counts[j] > counts[best] {
                        best = j;
                    }
                }
                col[t] = nb[best].0 as i64;
            }
        }
    }))
}

/// Inverse-distance weighted mean of each value column at every target, over
/// its `k` nearest source points within `max_distance` (weights
/// `1 / d^power`). Non-finite source values are skipped; a target at zero
/// distance from source points takes the mean of their values. NaN where no
/// neighbour with a finite value is left.
pub fn idw_values(source: &[Point], target: &[Point], values: &[Vec<f64>], k: usize, power: f64, max_distance: Option<f64>) -> Result<Vec<Vec<f64>>> {
    check_k(k, max_distance)?;
    check_power(power)?;
    for v in values {
        if v.len() != source.len() {
            return Err(Error::invalid(format!("value column has {} values for {} source points", v.len(), source.len())));
        }
    }
    let src = Source::new(source);
    Ok(fill_columns(target.len(), values.len(), f64::NAN, |start, cols| {
        for t in 0..cols.first().map_or(0, |c| c.len()) {
            let nb = src.knn(&target[start + t], k, max_distance);
            if nb.is_empty() {
                continue;
            }
            for (col, vals) in cols.iter_mut().zip(values) {
                col[t] = idw_mean(&nb, |i| vals[i], power);
            }
        }
    }))
}

fn check_power(power: f64) -> Result<()> {
    if !power.is_finite() || power < 0.0 {
        return Err(Error::invalid(format!("power must be a finite non-negative number, got {power}")));
    }
    Ok(())
}

/// Weighted mean over `(index, distance)` neighbours sorted by distance.
fn idw_mean(nb: &[(usize, f64)], value: impl Fn(usize) -> f64, power: f64) -> f64 {
    let (mut s, mut w) = (0.0, 0.0);
    // Coincident points: plain mean of their values.
    for &(i, _) in nb.iter().take_while(|e| e.1 == 0.0) {
        let v = value(i);
        if v.is_finite() {
            s += v;
            w += 1.0;
        }
    }
    if w > 0.0 {
        return s / w;
    }
    for &(i, d) in nb {
        let v = value(i);
        if v.is_finite() && d > 0.0 {
            let wi = d.powf(-power);
            s += wi * v;
            w += wi;
        }
    }
    if w > 0.0 {
        s / w
    } else {
        f64::NAN
    }
}

// --------------------------------------------------------------------- gridding

/// How [`grid`] interpolates between points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridMethod {
    /// Inverse distance weighting over the `k` nearest points (in x, y).
    Idw,
    /// Linear interpolation on the Delaunay triangulation.
    Tin,
    /// Natural-neighbour (Sibson) interpolation.
    Natural,
}

impl GridMethod {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "idw" => Ok(GridMethod::Idw),
            "tin" => Ok(GridMethod::Tin),
            "natural" => Ok(GridMethod::Natural),
            _ => Err(Error::invalid(format!("unknown grid method {name:?}; expected 'idw', 'tin' or 'natural'"))),
        }
    }
}

/// Settings for [`grid`].
#[derive(Debug, Clone, Copy)]
pub struct GridParams {
    pub method: GridMethod,
    /// Inverse-distance exponent (IDW only).
    pub power: f64,
    /// Neighbours per cell (IDW only).
    pub k: usize,
    /// Cells farther than this (in x, y) from every point are NaN.
    pub max_distance: Option<f64>,
}

/// Grid geometry `(nrows, ncols, xmin, ymin)` following
/// [`Raster::from_points`]: with no `bounds`, the corner is the points'
/// minimum snapped down to a multiple of `resolution`, and the grid covers
/// their maximum.
pub fn grid_geometry(xy: &[[f64; 2]], resolution: f64, bounds: Option<(f64, f64, f64, f64)>) -> Result<(usize, usize, f64, f64)> {
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(Error::invalid(format!("resolution must be a positive number, got {resolution}")));
    }
    let (xmin, ymin, xmax, ymax) = match bounds {
        Some(b) => {
            if !(b.0.is_finite() && b.1.is_finite() && b.2.is_finite() && b.3.is_finite()) || b.2 < b.0 || b.3 < b.1 {
                return Err(Error::invalid(format!("bounds must be finite (xmin, ymin, xmax, ymax) with xmax >= xmin and ymax >= ymin, got {b:?}")));
            }
            b
        }
        None => {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for p in xy {
                for d in 0..2 {
                    lo[d] = lo[d].min(p[d]);
                    hi[d] = hi[d].max(p[d]);
                }
            }
            if !lo[0].is_finite() {
                return Err(Error::invalid("cannot grid an empty point set without bounds"));
            }
            ((lo[0] / resolution).floor() * resolution, (lo[1] / resolution).floor() * resolution, hi[0], hi[1])
        }
    };
    let ncols = (((xmax - xmin) / resolution).floor() as usize + 1).max(1);
    let nrows = (((ymax - ymin) / resolution).floor() as usize + 1).max(1);
    Ok((nrows, ncols, xmin, ymin))
}

/// Points with finite x, y and value, sorted by (x, y), with points at the
/// same x, y merged into one whose value is their mean.
fn unique_points(points: &[Point], values: &[f64]) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..points.len()).filter(|&i| points[i][0].is_finite() && points[i][1].is_finite() && values[i].is_finite()).collect();
    idx.sort_by(|&a, &b| points[a][0].total_cmp(&points[b][0]).then(points[a][1].total_cmp(&points[b][1])).then(a.cmp(&b)));
    let mut xy: Vec<[f64; 2]> = Vec::with_capacity(idx.len());
    let mut val = Vec::with_capacity(idx.len());
    let mut i = 0;
    while i < idx.len() {
        let p = points[idx[i]];
        let mut j = i;
        let mut s = 0.0;
        while j < idx.len() && points[idx[j]][0] == p[0] && points[idx[j]][1] == p[1] {
            s += values[idx[j]];
            j += 1;
        }
        xy.push([p[0], p[1]]);
        val.push(s / (j - i) as f64);
        i = j;
    }
    (xy, val)
}

/// Interpolate `values` (one per point, located by the points' x, y) onto
/// the cell centres of a grid. Row 0 is at `ymin`, as for every [`Raster`].
///
/// Points with a non-finite x, y or value are ignored and points sharing an
/// x, y are merged (mean value). IDW gives every cell a value unless
/// `max_distance` is set; TIN and natural neighbour leave cells outside the
/// convex hull of the points NaN. With `max_distance`, cells whose nearest
/// point is farther than that are NaN for every method.
pub fn grid(points: &[Point], values: &[f64], resolution: f64, bounds: Option<(f64, f64, f64, f64)>, params: &GridParams) -> Result<Raster> {
    if values.len() != points.len() {
        return Err(Error::invalid(format!("{} values for {} points", values.len(), points.len())));
    }
    check_max_distance(params.max_distance)?;
    if params.method == GridMethod::Idw {
        check_k(params.k, None)?;
        check_power(params.power)?;
    }
    let (xy, val) = unique_points(points, values);
    let (nrows, ncols, xmin, ymin) = grid_geometry(&xy, resolution, bounds)?;
    let mut r = Raster::filled(nrows, ncols, xmin, ymin, resolution, f64::NAN);
    if xy.is_empty() {
        return Ok(r);
    }
    // Work relative to the centre of the data: triangulation predicates and
    // distances keep their precision with projected coordinates.
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &xy {
        for d in 0..2 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let c = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let local: Vec<Point> = xy.iter().map(|p| [p[0] - c[0], p[1] - c[1], 0.0]).collect();
    let centre = |row: usize, col: usize| -> Point { [xmin + (col as f64 + 0.5) * resolution - c[0], ymin + (row as f64 + 0.5) * resolution - c[1], 0.0] };
    let tree = (params.method == GridMethod::Idw || params.max_distance.is_some()).then(|| KdTree::new(&local));
    let too_far = |q: &Point| -> bool {
        match (params.max_distance, &tree) {
            (Some(m), Some(t)) => t.nearest(q).is_none_or(|(_, d)| d > m),
            _ => false,
        }
    };
    match params.method {
        GridMethod::Idw => {
            let tree = tree.as_ref().expect("tree");
            let k = params.k.min(local.len());
            r.data.par_chunks_mut(ncols).enumerate().for_each(|(row, out)| {
                for (col, cell) in out.iter_mut().enumerate() {
                    let q = centre(row, col);
                    let mut nb = tree.knn(&q, k);
                    if let Some(m) = params.max_distance {
                        nb.retain(|&(_, d)| d <= m);
                    }
                    *cell = idw_mean(&nb, |i| val[i], params.power);
                }
            });
        }
        GridMethod::Tin | GridMethod::Natural => {
            let vertices: Vec<Vertex> = local.iter().zip(&val).map(|(p, &v)| Vertex { x: p[0], y: p[1], value: v }).collect();
            let tri: Tin = Triangulation::bulk_load(vertices).map_err(|e| Error::invalid(format!("triangulation failed: {e:?}")))?;
            let natural = params.method == GridMethod::Natural;
            r.data.par_chunks_mut(ncols).enumerate().for_each(|(row, out)| {
                let bary = tri.barycentric();
                let nn = tri.natural_neighbor();
                for (col, cell) in out.iter_mut().enumerate() {
                    let q = centre(row, col);
                    if too_far(&q) {
                        continue;
                    }
                    let pos = Point2::new(q[0], q[1]);
                    let v = if natural { nn.interpolate(|h| h.data().value, pos) } else { bary.interpolate(|h| h.data().value, pos) };
                    *cell = v.unwrap_or(f64::NAN);
                }
            });
        }
    }
    Ok(r)
}

/// A triangulation vertex carrying the value to interpolate.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    x: f64,
    y: f64,
    value: f64,
}

impl HasPosition for Vertex {
    type Scalar = f64;

    fn position(&self) -> Point2<f64> {
        Point2::new(self.x, self.y)
    }
}

type Tin = DelaunayTriangulation<Vertex, (), (), (), BucketHint>;

/// Point-location hints from a fixed grid of buckets, each holding the
/// lowest-numbered vertex inside it (empty buckets take a neighbour's).
///
/// Unlike spade's own generators it keeps no state between queries, so a
/// lookup always starts from the same vertex and parallel queries give the
/// same results as serial ones.
#[derive(Debug, Default)]
struct BucketHint {
    origin: [f64; 2],
    size: f64,
    nx: usize,
    ny: usize,
    hint: Vec<usize>,
}

impl HintGenerator<f64> for BucketHint {
    fn get_hint(&self, position: Point2<f64>) -> FixedVertexHandle {
        if self.hint.is_empty() {
            return FixedVertexHandle::from_index(0);
        }
        let cell = |v: f64, o: f64, n: usize| -> usize {
            let f = ((v - o) / self.size).floor();
            if f.is_nan() || f < 0.0 {
                0
            } else {
                (f as usize).min(n - 1)
            }
        };
        let (i, j) = (cell(position.x, self.origin[0], self.nx), cell(position.y, self.origin[1], self.ny));
        FixedVertexHandle::from_index(self.hint[j * self.nx + i])
    }

    fn notify_vertex_lookup(&self, _: FixedVertexHandle) {}

    // Insertions happen only while bulk loading, after which spade rebuilds the
    // generator with `initialize_from_triangulation`.
    fn notify_vertex_inserted(&mut self, _: FixedVertexHandle, _: Point2<f64>) {}

    fn notify_vertex_removed(&mut self, _: Option<Point2<f64>>, _: FixedVertexHandle, _: Point2<f64>) {}

    fn initialize_from_triangulation<TR, V>(triangulation: &TR) -> Self
    where
        TR: Triangulation<Vertex = V>,
        V: HasPosition<Scalar = f64>,
    {
        let pos: Vec<Point2<f64>> = triangulation.vertices().map(|v| v.position()).collect();
        if pos.is_empty() {
            return BucketHint::default();
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in &pos {
            lo = [lo[0].min(p.x), lo[1].min(p.y)];
            hi = [hi[0].max(p.x), hi[1].max(p.y)];
        }
        // About two vertices per bucket over the bounding box.
        let area = ((hi[0] - lo[0]) * (hi[1] - lo[1])).max(f64::MIN_POSITIVE);
        let side = ((hi[0] - lo[0]).max(hi[1] - lo[1])).max(f64::MIN_POSITIVE);
        let mut size = (2.0 * area / pos.len() as f64).sqrt();
        if !(size.is_finite() && size > 0.0) || size < side / 4096.0 {
            size = side / 4096.0;
        }
        if size <= 0.0 || !size.is_finite() {
            size = 1.0;
        }
        let nx = (((hi[0] - lo[0]) / size).floor() as usize + 1).min(4097);
        let ny = (((hi[1] - lo[1]) / size).floor() as usize + 1).min(4097);
        let mut hint = vec![usize::MAX; nx * ny];
        for (k, p) in pos.iter().enumerate() {
            let i = (((p.x - lo[0]) / size).floor() as usize).min(nx - 1);
            let j = (((p.y - lo[1]) / size).floor() as usize).min(ny - 1);
            let b = &mut hint[j * nx + i];
            if *b == usize::MAX {
                *b = k;
            }
        }
        // Empty buckets inherit from their nearest filled bucket (breadth
        // first from the filled ones, in a fixed order).
        let mut queue: std::collections::VecDeque<usize> = (0..hint.len()).filter(|&b| hint[b] != usize::MAX).collect();
        while let Some(b) = queue.pop_front() {
            let (i, j) = (b % nx, b / nx);
            let nb = [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)];
            for (a, c) in nb {
                if a < nx && c < ny && hint[c * nx + a] == usize::MAX {
                    hint[c * nx + a] = hint[b];
                    queue.push_back(c * nx + a);
                }
            }
        }
        BucketHint { origin: lo, size, nx, ny, hint }
    }
}

// --------------------------------------------------------------------- sampling

/// How [`sample_raster`] reads a raster at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMethod {
    /// Bilinear between cell centres ([`Raster::sample`]).
    Bilinear,
    /// Value of the cell containing the point.
    Nearest,
}

impl SampleMethod {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "bilinear" => Ok(SampleMethod::Bilinear),
            "nearest" => Ok(SampleMethod::Nearest),
            _ => Err(Error::invalid(format!("unknown sampling method {name:?}; expected 'bilinear' or 'nearest'"))),
        }
    }
}

/// Raster values at each point's x, y: NaN for points outside the raster's
/// extent (`xmin <= x < xmax`, `ymin <= y < ymax`) or with non-finite
/// coordinates. Bilinear sampling holds the edge value in the outer half
/// cell and returns NaN when any of the four cells used is NaN.
pub fn sample_raster(raster: &Raster, points: &[Point], method: SampleMethod) -> Vec<f64> {
    points
        .par_iter()
        .map(|p| {
            let (row, col) = raster.cell_index(p[0], p[1]);
            if !(p[0].is_finite() && p[1].is_finite()) || !raster.in_bounds(row, col) {
                return f64::NAN;
            }
            match method {
                SampleMethod::Nearest => raster.get(row as usize, col as usize),
                SampleMethod::Bilinear => raster.sample(p[0], p[1]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic scattered points in `[0, w) x [0, h)`.
    fn scatter(n: usize, w: f64, h: f64) -> Vec<Point> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| [next() * w, next() * h, 0.0]).collect()
    }

    fn params(method: GridMethod) -> GridParams {
        GridParams { method, power: 2.0, k: 12, max_distance: None }
    }

    #[test]
    fn tin_and_natural_reproduce_a_plane() {
        let pts = scatter(400, 20.0, 10.0);
        let plane = |x: f64, y: f64| 3.0 + 0.25 * x - 0.5 * y;
        let v: Vec<f64> = pts.iter().map(|p| plane(p[0], p[1])).collect();
        for m in [GridMethod::Tin, GridMethod::Natural] {
            let r = grid(&pts, &v, 0.5, Some((0.0, 0.0, 20.0, 10.0)), &params(m)).unwrap();
            let mut inside = 0;
            for row in 0..r.nrows {
                for col in 0..r.ncols {
                    let z = r.get(row, col);
                    if z.is_finite() {
                        let (x, y) = r.cell_center(row, col);
                        assert!((z - plane(x, y)).abs() < 1e-9, "{m:?} at {x},{y}: {z}");
                        inside += 1;
                    }
                }
            }
            assert!(inside > r.data.len() / 2);
        }
    }

    #[test]
    fn tin_is_nan_outside_the_hull_and_idw_is_bounded() {
        let pts = vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let v = vec![1.0, 2.0, 5.0];
        let r = grid(&pts, &v, 1.0, Some((0.0, 0.0, 4.0, 4.0)), &params(GridMethod::Tin)).unwrap();
        assert!(r.get(0, 0).is_finite());
        assert!(r.get(3, 3).is_nan());
        let r = grid(&pts, &v, 1.0, Some((-10.0, -10.0, 10.0, 10.0)), &params(GridMethod::Idw)).unwrap();
        assert!(r.data.iter().all(|&z| (1.0..=5.0).contains(&z)));
        let far = GridParams { max_distance: Some(1.0), ..params(GridMethod::Idw) };
        let r = grid(&pts, &v, 1.0, Some((-10.0, -10.0, 10.0, 10.0)), &far).unwrap();
        assert!(r.get(0, 0).is_nan());
        assert!(r.data.iter().any(|z| z.is_finite()));
    }

    #[test]
    fn grid_geometry_matches_from_points() {
        let pts = scatter(50, 7.3, 3.1);
        let xy: Vec<[f64; 2]> = pts.iter().map(|p| [p[0], p[1]]).collect();
        let (nrows, ncols, xmin, ymin) = grid_geometry(&xy, 0.4, None).unwrap();
        let r = Raster::from_points(pts.iter().map(|p| (p[0], p[1])), pts.iter().map(|_| 0.0), 0.4, crate::raster::Reducer::Count, None, 0.0).unwrap();
        assert_eq!((nrows, ncols, xmin, ymin), (r.nrows, r.ncols, r.xmin, r.ymin));
        assert!(grid_geometry(&[], 1.0, None).is_err());
        assert!(grid_geometry(&xy, 0.0, None).is_err());
    }

    #[test]
    fn duplicates_are_averaged() {
        let pts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let v = vec![1.0, 3.0, 2.0, 2.0];
        let r = grid(&pts, &v, 0.5, Some((-0.25, -0.25, 0.0, 0.0)), &params(GridMethod::Tin)).unwrap();
        // The single cell centre is at (0, 0).
        assert!((r.get(0, 0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_and_majority() {
        let src = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
        let tgt = vec![[0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [50.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
        assert_eq!(nearest_indices(&src, &tgt, None).unwrap(), vec![1, 0, 3, -1]);
        assert_eq!(nearest_indices(&src, &tgt, Some(5.0)).unwrap(), vec![1, 0, -1, -1]);
        // Labels: 7, 5, 5, 7 (and 9 on the NaN point, never used).
        let lab = label_keys(&Attr::I64(vec![7, 5, 5, 7, 9]));
        let m = majority_indices(&src, &tgt, std::slice::from_ref(&lab), 3, None).unwrap();
        // Target 0: neighbours 1 (5), 2 (5), 0 (7): majority 5, nearest is 1.
        // Target 1: neighbours 0 (7), 1 (5), 2 (5): majority 5, nearest is 1.
        assert_eq!(m[0][..2], [1, 1]);
        // A tie (k = 2 around target 1: 7 and 5) goes to the nearest point.
        let m = majority_indices(&src, &tgt, &[lab], 2, None).unwrap();
        assert_eq!(m[0][1], 0);
        assert_eq!(m[0][3], -1);
    }

    #[test]
    fn label_keys_identify_equal_floats() {
        let k = label_keys(&Attr::F64(vec![0.0, -0.0, f64::NAN, -f64::NAN, 1.0]));
        assert_eq!(k[0], k[1]);
        assert_eq!(k[2], k[3]);
        assert_ne!(k[0], k[4]);
    }

    #[test]
    fn idw_is_exact_at_data_and_bounded() {
        let src = scatter(200, 5.0, 5.0);
        let v: Vec<f64> = src.iter().map(|p| (p[0] * 1.3).sin() + p[1]).collect();
        let out = idw_values(&src, &src, std::slice::from_ref(&v), 8, 2.0, None).unwrap();
        for (a, b) in out[0].iter().zip(&v) {
            assert!((a - b).abs() < 1e-12);
        }
        let tgt = scatter(300, 6.0, 6.0);
        let out = idw_values(&src, &tgt, std::slice::from_ref(&v), 8, 2.0, None).unwrap();
        let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &b| (a.0.min(b), a.1.max(b)));
        assert!(out[0].iter().all(|&z| z >= lo && z <= hi));
        assert!(idw_values(&src, &tgt, std::slice::from_ref(&v), 0, 2.0, None).is_err());
        assert!(idw_values(&src, &tgt, &[v], 8, -1.0, None).is_err());
        let empty = idw_values(&[], &tgt, &[vec![]], 8, 2.0, None).unwrap();
        assert!(empty[0].iter().all(|z| z.is_nan()));
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let pts = scatter(3000, 30.0, 30.0);
        let v: Vec<f64> = pts.iter().map(|p| (p[0] * 0.3).sin() * (p[1] * 0.2).cos()).collect();
        let run = |threads: usize, m: GridMethod| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| grid(&pts, &v, 0.25, None, &params(m)).unwrap().data)
        };
        for m in [GridMethod::Idw, GridMethod::Tin, GridMethod::Natural] {
            let a = run(1, m);
            let b = run(4, m);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{m:?}");
        }
    }

    #[test]
    fn sampling_is_nan_outside() {
        let mut r = Raster::filled(2, 2, 0.0, 0.0, 1.0, 0.0);
        r.data = vec![1.0, 2.0, 3.0, 4.0];
        let p = vec![[0.5, 0.5, 0.0], [1.0, 1.0, 0.0], [1.9, 1.9, 0.0], [2.0, 0.5, 0.0], [-0.1, 0.5, 0.0]];
        let b = sample_raster(&r, &p, SampleMethod::Bilinear);
        assert_eq!(b[0], 1.0);
        assert_eq!(b[1], 2.5);
        assert_eq!(b[2], 4.0);
        assert!(b[3].is_nan() && b[4].is_nan());
        let n = sample_raster(&r, &p, SampleMethod::Nearest);
        assert_eq!(n[..3], [1.0, 4.0, 4.0]);
    }
}
