// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Point-in-polygon tests for many points against many polygons.
//!
//! A polygon is closed: a point on an edge or a vertex of the exterior ring
//! is inside, and so is a point on the boundary of a hole (only the open
//! interior of a hole is removed). The decision uses the exact orientation
//! predicate of Shewchuk (the `robust` crate), so a point on an edge is
//! recognised as such whatever the edge's slope, and a point shared by two
//! adjacent polygons belongs to both.
//!
//! Speed comes from two indexes: a uniform grid over the polygons' bounding
//! boxes, so a point is only tested against the polygons whose box holds it,
//! and horizontal bands over each ring's edges, so a test only visits the
//! edges that cross the point's band.

use rayon::prelude::*;
use robust::{orient2d, Coord};

use super::CHUNK;
use crate::error::{Error, Result};
use crate::Point;

/// A 2-D coordinate.
pub type Xy = [f64; 2];

/// A polygon: one exterior ring and any number of holes.
///
/// Rings may be given open or closed (a repeated first vertex is dropped),
/// in either orientation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polygon {
    pub exterior: Vec<Xy>,
    pub holes: Vec<Vec<Xy>>,
}

/// A set of polygons treated as one shape (a point is inside if it is inside
/// any part). A multipolygon with no parts contains nothing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MultiPolygon {
    pub parts: Vec<Polygon>,
}

impl From<Polygon> for MultiPolygon {
    fn from(p: Polygon) -> Self {
        MultiPolygon { parts: vec![p] }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    Outside,
    Boundary,
    Inside,
}

/// Axis-aligned box `[xmin, ymin, xmax, ymax]`.
type BBox = [f64; 4];

fn bbox_union(a: BBox, b: BBox) -> BBox {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

#[inline]
fn bbox_contains(b: &BBox, x: f64, y: f64) -> bool {
    x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]
}

/// Index of the band holding `v` on `n` bands of width `h` from `lo`. The map
/// is monotone in `v`, so an interval's bands always include those of the
/// values inside it.
#[inline]
fn band(v: f64, lo: f64, h: f64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let b = ((v - lo) / h).floor();
    if b <= 0.0 {
        0
    } else {
        (b as usize).min(n - 1)
    }
}

/// A ring with its edges bucketed into horizontal bands.
#[derive(Debug, Clone)]
struct Ring {
    vertices: Vec<Xy>,
    bbox: BBox,
    band_h: f64,
    band_start: Vec<u32>,
    band_edges: Vec<u32>,
}

impl Ring {
    fn new(raw: &[Xy], what: &str) -> Result<Self> {
        let mut vertices: Vec<Xy> = Vec::with_capacity(raw.len());
        for &v in raw {
            if !v[0].is_finite() || !v[1].is_finite() {
                return Err(Error::invalid(format!("{what} has a non-finite vertex ({}, {})", v[0], v[1])));
            }
            if vertices.last() != Some(&v) {
                vertices.push(v);
            }
        }
        while vertices.len() > 1 && vertices.first() == vertices.last() {
            vertices.pop();
        }
        if vertices.len() < 3 {
            return Err(Error::invalid(format!("{what} has {} distinct vertices; a ring needs at least 3", vertices.len())));
        }
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for v in &vertices {
            bbox = bbox_union(bbox, [v[0], v[1], v[0], v[1]]);
        }
        let n = vertices.len();
        let height = bbox[3] - bbox[1];
        let nbands = if height > 0.0 { (n / 4).clamp(1, 1 << 16) } else { 1 };
        let band_h = if nbands > 1 { height / nbands as f64 } else { 1.0 };
        let edge_bands = |i: usize| {
            let (a, b) = (vertices[i], vertices[(i + 1) % n]);
            (band(a[1].min(b[1]), bbox[1], band_h, nbands), band(a[1].max(b[1]), bbox[1], band_h, nbands))
        };
        // CSR of edge indices per band.
        let mut counts = vec![0u32; nbands + 1];
        for i in 0..n {
            let (lo, hi) = edge_bands(i);
            for c in &mut counts[lo + 1..=hi + 1] {
                *c += 1;
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let mut fill = counts.clone();
        let mut band_edges = vec![0u32; counts[nbands] as usize];
        for i in 0..n {
            let (lo, hi) = edge_bands(i);
            for f in &mut fill[lo..=hi] {
                band_edges[*f as usize] = i as u32;
                *f += 1;
            }
        }
        Ok(Ring { vertices, bbox, band_h, band_start: counts, band_edges })
    }

    fn locate(&self, x: f64, y: f64) -> Location {
        if !bbox_contains(&self.bbox, x, y) {
            return Location::Outside;
        }
        let nbands = self.band_start.len() - 1;
        let b = band(y, self.bbox[1], self.band_h, nbands);
        let n = self.vertices.len();
        let p = Coord { x, y };
        let mut inside = false;
        for &e in &self.band_edges[self.band_start[b] as usize..self.band_start[b + 1] as usize] {
            let e = e as usize;
            let (a, c) = (self.vertices[e], self.vertices[(e + 1) % n]);
            let (ca, cc) = (Coord { x: a[0], y: a[1] }, Coord { x: c[0], y: c[1] });
            if (a[1] > y) != (c[1] > y) {
                // The edge straddles the horizontal line through p (half-open
                // in y), so it is not horizontal and p lies on it exactly when
                // the three points are collinear.
                let o = orient2d(ca, cc, p);
                if o == 0.0 {
                    return Location::Boundary;
                }
                if (o > 0.0) == (c[1] > a[1]) {
                    inside = !inside;
                }
            } else if y >= a[1].min(c[1]) && y <= a[1].max(c[1]) && x >= a[0].min(c[0]) && x <= a[0].max(c[0]) && orient2d(ca, cc, p) == 0.0 {
                return Location::Boundary;
            }
        }
        if inside {
            Location::Inside
        } else {
            Location::Outside
        }
    }
}

#[derive(Debug, Clone)]
struct Part {
    exterior: Ring,
    holes: Vec<Ring>,
}

impl Part {
    fn contains(&self, x: f64, y: f64) -> bool {
        match self.exterior.locate(x, y) {
            Location::Outside => false,
            Location::Boundary => true,
            Location::Inside => self.holes.iter().all(|h| h.locate(x, y) != Location::Inside),
        }
    }
}

/// Polygons prepared for fast point-in-polygon queries.
#[derive(Debug, Clone)]
pub struct PolygonIndex {
    features: Vec<(BBox, Vec<Part>)>,
    grid_bbox: BBox,
    nx: usize,
    ny: usize,
    cell_w: f64,
    cell_h: f64,
    cell_start: Vec<u32>,
    cell_features: Vec<u32>,
}

impl PolygonIndex {
    /// Prepare `features` (each a multipolygon) for queries.
    ///
    /// # Errors
    /// A ring with a non-finite vertex or fewer than three distinct
    /// vertices; the message names the feature, part and ring.
    pub fn new(features: &[MultiPolygon]) -> Result<Self> {
        let prepared: Vec<(BBox, Vec<Part>)> = features
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                let mut parts = Vec::with_capacity(f.parts.len());
                for (pi, p) in f.parts.iter().enumerate() {
                    let exterior = Ring::new(&p.exterior, &format!("polygon {fi}, part {pi}, exterior ring"))?;
                    bbox = bbox_union(bbox, exterior.bbox);
                    let holes = p.holes.iter().enumerate().map(|(hi, h)| Ring::new(h, &format!("polygon {fi}, part {pi}, hole {hi}"))).collect::<Result<Vec<_>>>()?;
                    parts.push(Part { exterior, holes });
                }
                Ok((bbox, parts))
            })
            .collect::<Result<_>>()?;
        let mut grid_bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut nonempty = 0usize;
        for (b, parts) in &prepared {
            if !parts.is_empty() {
                grid_bbox = bbox_union(grid_bbox, *b);
                nonempty += 1;
            }
        }
        let side = ((4 * nonempty) as f64).sqrt().ceil().clamp(1.0, 1024.0) as usize;
        let (w, h) = (grid_bbox[2] - grid_bbox[0], grid_bbox[3] - grid_bbox[1]);
        let nx = if w > 0.0 { side } else { 1 };
        let ny = if h > 0.0 { side } else { 1 };
        let cell_w = if nx > 1 { w / nx as f64 } else { 1.0 };
        let cell_h = if ny > 1 { h / ny as f64 } else { 1.0 };
        let cells_of = |b: &BBox| {
            let (x0, x1) = (band(b[0], grid_bbox[0], cell_w, nx), band(b[2], grid_bbox[0], cell_w, nx));
            let (y0, y1) = (band(b[1], grid_bbox[1], cell_h, ny), band(b[3], grid_bbox[1], cell_h, ny));
            (x0, x1, y0, y1)
        };
        let mut counts = vec![0u32; nx * ny + 1];
        for (b, parts) in &prepared {
            if parts.is_empty() {
                continue;
            }
            let (x0, x1, y0, y1) = cells_of(b);
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    counts[cy * nx + cx + 1] += 1;
                }
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let mut fill = counts.clone();
        let mut cell_features = vec![0u32; counts[nx * ny] as usize];
        // Features are visited in order, so each cell lists them ascending.
        for (fi, (b, parts)) in prepared.iter().enumerate() {
            if parts.is_empty() {
                continue;
            }
            let (x0, x1, y0, y1) = cells_of(b);
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    let c = cy * nx + cx;
                    cell_features[fill[c] as usize] = fi as u32;
                    fill[c] += 1;
                }
            }
        }
        Ok(PolygonIndex { features: prepared, grid_bbox, nx, ny, cell_w, cell_h, cell_start: counts, cell_features })
    }

    /// Number of features indexed.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Index of the first feature containing `(x, y)`, if any.
    pub fn locate(&self, x: f64, y: f64) -> Option<usize> {
        if !x.is_finite() || !y.is_finite() || !bbox_contains(&self.grid_bbox, x, y) {
            return None;
        }
        let cx = band(x, self.grid_bbox[0], self.cell_w, self.nx);
        let cy = band(y, self.grid_bbox[1], self.cell_h, self.ny);
        let c = cy * self.nx + cx;
        self.cell_features[self.cell_start[c] as usize..self.cell_start[c + 1] as usize].iter().map(|&f| f as usize).find(|&f| {
            let (b, parts) = &self.features[f];
            bbox_contains(b, x, y) && parts.iter().any(|p| p.contains(x, y))
        })
    }

    /// For each point, the index of the first feature whose polygons contain
    /// its x, y, or -1. Points with a non-finite x or y get -1.
    pub fn feature_of(&self, points: &[Point]) -> Vec<i64> {
        let mut out = vec![-1i64; points.len()];
        out.par_chunks_mut(CHUNK).zip(points.par_chunks(CHUNK)).for_each(|(o, p)| {
            for (o, p) in o.iter_mut().zip(p) {
                *o = self.locate(p[0], p[1]).map_or(-1, |f| f as i64);
            }
        });
        out
    }

    /// For each point, whether any feature contains its x, y.
    pub fn contains(&self, points: &[Point]) -> Vec<bool> {
        let mut out = vec![false; points.len()];
        out.par_chunks_mut(CHUNK).zip(points.par_chunks(CHUNK)).for_each(|(o, p)| {
            for (o, p) in o.iter_mut().zip(p) {
                *o = self.locate(p[0], p[1]).is_some();
            }
        });
        out
    }
}

/// Whether the ring `(x, y)` pairs enclose `p` (boundary counts as inside).
/// Used to attach shapefile holes to their exterior ring.
pub(crate) fn ring_contains(ring: &[Xy], p: Xy) -> bool {
    Ring::new(ring, "ring").map(|r| r.locate(p[0], p[1]) != Location::Outside).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, s: f64) -> Vec<Xy> {
        vec![[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]]
    }

    fn index(polys: Vec<Polygon>) -> PolygonIndex {
        PolygonIndex::new(&polys.into_iter().map(MultiPolygon::from).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn square_interior_boundary_exterior() {
        let idx = index(vec![Polygon { exterior: square(0.0, 0.0, 1.0), holes: vec![] }]);
        assert_eq!(idx.locate(0.5, 0.5), Some(0));
        assert_eq!(idx.locate(1.5, 0.5), None);
        // Edges and vertices are inside.
        for (x, y) in [(0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)] {
            assert_eq!(idx.locate(x, y), Some(0), "({x}, {y})");
        }
        assert_eq!(idx.locate(1.0 + 1e-12, 0.5), None);
        assert_eq!(idx.locate(f64::NAN, 0.5), None);
    }

    #[test]
    fn hole_interior_removed_boundary_kept() {
        let idx = index(vec![Polygon { exterior: square(0.0, 0.0, 4.0), holes: vec![square(1.0, 1.0, 2.0)] }]);
        assert_eq!(idx.locate(2.0, 2.0), None);
        assert_eq!(idx.locate(1.0, 2.0), Some(0));
        assert_eq!(idx.locate(0.5, 2.0), Some(0));
        assert_eq!(idx.locate(3.5, 3.5), Some(0));
    }

    #[test]
    fn slanted_edge_points_are_exact() {
        // Points on a slanted edge between (0,0) and (3,1): x = 3k, y = k.
        let idx = index(vec![Polygon { exterior: vec![[0.0, 0.0], [3.0, 1.0], [0.0, 2.0]], holes: vec![] }]);
        for k in [0.1, 0.25, 1.0 / 3.0, 0.7] {
            let (x, y) = (3.0 * k, k);
            if orient2d(Coord { x: 0.0, y: 0.0 }, Coord { x: 3.0, y: 1.0 }, Coord { x, y }) == 0.0 {
                assert_eq!(idx.locate(x, y), Some(0));
            }
        }
        assert_eq!(idx.locate(1.5, 0.5), Some(0));
    }

    #[test]
    fn shared_edge_belongs_to_both() {
        let idx = index(vec![Polygon { exterior: square(0.0, 0.0, 1.0), holes: vec![] }, Polygon { exterior: square(1.0, 0.0, 1.0), holes: vec![] }]);
        assert_eq!(idx.locate(1.0, 0.5), Some(0));
        let pts = [[1.5, 0.5, 0.0], [0.5, 0.5, 0.0], [3.0, 0.5, 0.0]];
        assert_eq!(idx.feature_of(&pts), vec![1, 0, -1]);
    }

    #[test]
    fn concave_and_closed_rings() {
        // A "U" shape given as a closed ring, clockwise.
        let u = vec![[0.0, 0.0], [0.0, 3.0], [1.0, 3.0], [1.0, 1.0], [2.0, 1.0], [2.0, 3.0], [3.0, 3.0], [3.0, 0.0], [0.0, 0.0]];
        let idx = index(vec![Polygon { exterior: u, holes: vec![] }]);
        assert_eq!(idx.locate(1.5, 2.0), None);
        assert_eq!(idx.locate(0.5, 2.0), Some(0));
        assert_eq!(idx.locate(1.5, 0.5), Some(0));
        // A ray through the vertex (1, 1) must not double count.
        assert_eq!(idx.locate(0.5, 1.0), Some(0));
        assert_eq!(idx.locate(-0.5, 1.0), None);
        assert_eq!(idx.locate(1.5, 1.0), Some(0));
    }

    #[test]
    fn many_vertex_circle_matches_radius() {
        let n = 10_000;
        let ring: Vec<Xy> = (0..n).map(|i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            [10.0 * t.cos(), 10.0 * t.sin()]
        }).collect();
        let idx = index(vec![Polygon { exterior: ring, holes: vec![] }]);
        let mut s = 1u64;
        for _ in 0..20_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = (s % 30_000) as f64 / 1000.0 - 15.0;
            let y = ((s >> 20) % 30_000) as f64 / 1000.0 - 15.0;
            let r = (x * x + y * y).sqrt();
            // The inscribed polygon departs from the circle by < 10 * (1 - cos(pi / n)).
            if (r - 10.0).abs() > 1e-5 {
                assert_eq!(idx.locate(x, y).is_some(), r < 10.0, "({x}, {y})");
            }
        }
    }

    #[test]
    fn bad_rings_are_rejected() {
        let e = PolygonIndex::new(&[Polygon { exterior: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], holes: vec![] }.into()]).unwrap_err();
        assert!(e.to_string().contains("polygon 0, part 0, exterior ring has 2 distinct vertices"), "{e}");
        let e = PolygonIndex::new(&[Polygon { exterior: vec![[0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]], holes: vec![] }.into()]).unwrap_err();
        assert!(e.to_string().contains("non-finite"), "{e}");
    }

    #[test]
    fn empty_features_contain_nothing() {
        let idx = PolygonIndex::new(&[MultiPolygon::default()]).unwrap();
        assert_eq!(idx.feature_of(&[[0.0, 0.0, 0.0]]), vec![-1]);
        let idx = PolygonIndex::new(&[]).unwrap();
        assert!(idx.contains(&[[0.0, 0.0, 0.0]]) == vec![false]);
    }
}
