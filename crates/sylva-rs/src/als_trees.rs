// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Individual trees from airborne lidar: tree tops, crowns and labelled
//! points, for one cloud or a whole catalogue.
//!
//! The algorithms are those of lidR (Roussel et al. 2020), whose
//! implementations are followed line by line where the papers leave a
//! detail open:
//!
//! - **Tree tops** ([`local_maxima_raster`], [`local_maxima_points`]): the
//!   local maximum filter `lmf` (Popescu and Wynne 2004). A site (a CHM
//!   cell centre or a point) at or above `hmin` is a tree top when no other
//!   site within its window is higher. The window is a circle of diameter
//!   `ws` (or a square of side `ws`) centred on the site, and `ws` may
//!   depend on the site's height ([`Window`]). Of equal-height maxima that
//!   lie in each other's windows only the first is kept; lidR keeps the
//!   first it happens to tag, here the first in order of x, then y.
//! - **Marker-controlled watershed** ([`watershed`]): Meyer's flooding
//!   (Meyer and Beucher 1990; Meyer 1991) of the CHM from the tree tops,
//!   highest cells first, over 8-connected cells higher than `th_tree`,
//!   without watershed lines.
//! - **Dalponte and Coomes (2016)** ([`dalponte2016`]): seeded region
//!   growing on the CHM, a port of lidR's `C_dalponte2016` (thresholds
//!   `th_tree`, `th_seed`, `th_cr`, the 5 % rule above the seed and the
//!   `max_cr` window, with its sweep order and bookkeeping).
//! - **Li et al. (2012)** ([`li2012`]): point-based region growing from the
//!   highest point down, a port of lidR's `LAS::segment_trees`.
//!
//! [`segment_cloud`] runs a whole segmentation on one cloud and
//! [`catalog_trees`] on a catalogue through the chunk engine of
//! [`crate::als`]: each tree belongs to the chunk that holds its top (a
//! top from a CHM cell to the chunk whose core is nearest the cell centre,
//! as [`crate::als::mosaic`] assigns cells; a point top to the chunk whose
//! core holds the point), is segmented there with the buffer's points, and
//! gets an id from the order of all tops in the catalogue (by x, then y),
//! so ids and crowns do not depend on the chunks or the threads.
//!
//! References: Dalponte, M. and Coomes, D. A. (2016) Methods in Ecology
//! and Evolution 7, 1236-1245. Li, W., Guo, Q., Jakubowski, M. K. and
//! Kelly, M. (2012) Photogrammetric Engineering & Remote Sensing 78(1),
//! 75-84. Meyer, F. and Beucher, S. (1990) Journal of Visual Communication
//! and Image Representation 1(1), 21-46. Meyer, F. (1991) 8e congrès
//! AFCET, Lyon-Villeurbanne, 847-857. Popescu, S. C. and Wynne, R. H.
//! (2004) Photogrammetric Engineering & Remote Sensing 70(5), 589-604.
//! Duckham, M., Kulik, L., Worboys, M. and Galton, A. (2008) Pattern
//! Recognition 41(10), 3224-3236.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};

use crate::als::{catalog_grid, chunk_bounds, output_path, run, workers_for, write_like, Catalog, Chunk, BYTES_PER_POINT};
use crate::als_ops::{chunk_heights, Heights, RunOptions};
use crate::error::{Error, Result};
use crate::ground;
use crate::pointcloud::Attr;
use crate::raster::Raster;
use crate::trees::{convex_hull, polygon_area};
use crate::Point;

// ------------------------------------------------------------------ windows

/// Window size `ws` (m) of the local maximum filter at a site.
#[derive(Debug, Clone, PartialEq)]
pub enum Window {
    /// The same everywhere.
    Fixed(f64),
    /// `clamp(intercept + slope * h, min, max)` for a site of height `h`.
    Linear { intercept: f64, slope: f64, min: f64, max: f64 },
    /// `values[k]` at height `k * step`, linear in between, the end values
    /// beyond: a sampled user function.
    Table { step: f64, values: Vec<f64> },
    /// One value per site (per cell, row-major, for a raster; per point
    /// for points).
    PerSite(Vec<f64>),
}

impl Window {
    fn check(&self) -> Result<()> {
        let bad = |v: f64| !(v.is_finite() && v > 0.0);
        match self {
            Window::Fixed(w) if bad(*w) => Err(Error::invalid(format!("the window size must be a positive number of metres, got {w}"))),
            Window::Linear { intercept, slope, min, max } => {
                if !(intercept.is_finite() && slope.is_finite()) || bad(*min) || !(max.is_finite() && max >= min) {
                    return Err(Error::invalid(format!("a linear window needs finite intercept and slope and 0 < min <= max, got intercept {intercept}, slope {slope}, min {min}, max {max}")));
                }
                Ok(())
            }
            Window::Table { step, values } => {
                if bad(*step) || values.is_empty() {
                    return Err(Error::invalid("a window table needs a positive step and at least one value"));
                }
                if let Some(v) = values.iter().find(|v| bad(**v)) {
                    return Err(Error::invalid(format!("the window function must give positive sizes, got {v}")));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `ws` for site `i` of height `h`.
    pub fn at(&self, i: usize, h: f64) -> f64 {
        match self {
            Window::Fixed(w) => *w,
            Window::Linear { intercept, slope, min, max } => (intercept + slope * h).clamp(*min, *max),
            Window::Table { step, values } => {
                let f = (h / step).max(0.0);
                let k = f.floor() as usize;
                if k + 1 >= values.len() {
                    return values[values.len() - 1];
                }
                let t = f - k as f64;
                values[k] + t * (values[k + 1] - values[k])
            }
            Window::PerSite(v) => v[i],
        }
    }

    /// The largest `ws` the window can give (for sites up to `hmax` for a table).
    pub fn largest(&self) -> f64 {
        match self {
            Window::Fixed(w) => *w,
            Window::Linear { max, .. } => *max,
            Window::Table { values, .. } | Window::PerSite(values) => values.iter().copied().filter(|v| v.is_finite()).fold(0.0, f64::max),
        }
    }
}

/// Shape of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// A disc of diameter `ws`.
    Circular,
    /// A square of side `ws`, aligned with the axes.
    Square,
}

impl Shape {
    pub fn parse(s: &str) -> Result<Shape> {
        match s {
            "circular" => Ok(Shape::Circular),
            "square" => Ok(Shape::Square),
            _ => Err(Error::invalid(format!("unknown window shape {s:?}; expected 'circular' or 'square'"))),
        }
    }

    #[inline]
    fn contains(self, dx: f64, dy: f64, r: f64) -> bool {
        match self {
            Shape::Circular => dx * dx + dy * dy <= r * r,
            Shape::Square => dx.abs() <= r && dy.abs() <= r,
        }
    }
}

fn site_window(window: &Window, i: usize, h: f64) -> Result<f64> {
    let w = window.at(i, h);
    if !(w.is_finite() && w > 0.0) {
        return Err(Error::invalid(format!("the window size at a site {h} m high is {w}; it must be a positive number of metres")));
    }
    Ok(w)
}

// ------------------------------------------------------------------ local maxima

/// Tree tops found by the local maximum filter on a CHM: the cells (row-major
/// indices), in order of x, then y.
///
/// A cell is a top when its value is at least `hmin` and no cell whose centre
/// lies in its window is higher; NaN cells take no part. Cell centre offsets
/// are whole numbers of cells, so the windows do not depend on where the
/// raster starts.
pub fn local_maxima_raster(chm: &Raster, window: &Window, hmin: f64, shape: Shape) -> Result<Vec<usize>> {
    window.check()?;
    if let Window::PerSite(v) = window {
        if v.len() != chm.data.len() {
            return Err(Error::invalid(format!("{} window sizes for {} cells", v.len(), chm.data.len())));
        }
    }
    let (nr, nc, res) = (chm.nrows as i64, chm.ncols as i64, chm.resolution);
    let v = &chm.data;
    let cand: Vec<Result<Option<(usize, i64)>>> = (0..v.len())
        .into_par_iter()
        .map(|i| {
            let z = v[i];
            if !(z.is_finite() && z >= hmin) {
                return Ok(None);
            }
            let r = site_window(window, i, z)? / 2.0;
            let (row, col) = ((i as i64) / nc, (i as i64) % nc);
            let k = (r / res + 1e-9).floor() as i64;
            let inside = |dr: i64, dc: i64| shape.contains(dc as f64 * res, dr as f64 * res, r);
            // Quick rejection by the 8 neighbours when they are in the window.
            if inside(1, 1) {
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let (rr, cc) = (row + dr, col + dc);
                        if (dr != 0 || dc != 0) && rr >= 0 && rr < nr && cc >= 0 && cc < nc && v[(rr * nc + cc) as usize] > z {
                            return Ok(None);
                        }
                    }
                }
            }
            for dr in -k..=k {
                let rr = row + dr;
                if rr < 0 || rr >= nr {
                    continue;
                }
                for dc in -k..=k {
                    let cc = col + dc;
                    if cc < 0 || cc >= nc || !inside(dr, dc) {
                        continue;
                    }
                    if v[(rr * nc + cc) as usize] > z {
                        return Ok(None);
                    }
                }
            }
            Ok(Some((i, k)))
        })
        .collect();
    let mut cand: Vec<(usize, i64)> = cand.into_iter().collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();
    // x, then y: column, then row.
    cand.sort_by_key(|&(i, _)| (i % chm.ncols, i / chm.ncols));
    let mut is_top = vec![false; v.len()];
    let mut tops = Vec::new();
    for (i, k) in cand {
        let z = v[i];
        let r = window.at(i, z) / 2.0;
        let (row, col) = ((i as i64) / nc, (i as i64) % nc);
        let mut tie = false;
        'outer: for dr in -k..=k {
            let rr = row + dr;
            if rr < 0 || rr >= nr {
                continue;
            }
            for dc in -k..=k {
                let cc = col + dc;
                if cc < 0 || cc >= nc || (dr == 0 && dc == 0) || !shape.contains(dc as f64 * res, dr as f64 * res, r) {
                    continue;
                }
                let j = (rr * nc + cc) as usize;
                if is_top[j] && v[j] == z {
                    tie = true;
                    break 'outer;
                }
            }
        }
        if !tie {
            is_top[i] = true;
            tops.push(i);
        }
    }
    Ok(tops)
}

/// A static bucket grid over 2-D sites (compressed rows).
struct Buckets {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    start: Vec<u32>,
    idx: Vec<u32>,
}

impl Buckets {
    fn new(xy: &[[f64; 2]], sites: &[usize], cell: f64) -> Buckets {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &i in sites {
            for k in 0..2 {
                lo[k] = lo[k].min(xy[i][k]);
                hi[k] = hi[k].max(xy[i][k]);
            }
        }
        if sites.is_empty() {
            return Buckets { x0: 0.0, y0: 0.0, cell, nx: 1, ny: 1, start: vec![0, 0], idx: vec![] };
        }
        // At most about 16 million cells.
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let cell = cell.max(span / 4000.0).max(1e-6);
        let nx = ((hi[0] - lo[0]) / cell).floor() as usize + 1;
        let ny = ((hi[1] - lo[1]) / cell).floor() as usize + 1;
        let mut b = Buckets { x0: lo[0], y0: lo[1], cell, nx, ny, start: vec![0; nx * ny + 1], idx: vec![0; sites.len()] };
        for &i in sites {
            let c = b.cell_of(xy[i][0], xy[i][1]);
            b.start[c + 1] += 1;
        }
        for k in 1..b.start.len() {
            b.start[k] += b.start[k - 1];
        }
        let mut fill = b.start.clone();
        for &i in sites {
            let c = b.cell_of(xy[i][0], xy[i][1]);
            b.idx[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        b
    }

    #[inline]
    fn col(&self, x: f64) -> i64 {
        ((x - self.x0) / self.cell).floor() as i64
    }

    #[inline]
    fn row(&self, y: f64) -> i64 {
        ((y - self.y0) / self.cell).floor() as i64
    }

    fn cell_of(&self, x: f64, y: f64) -> usize {
        let c = self.col(x).clamp(0, self.nx as i64 - 1) as usize;
        let r = self.row(y).clamp(0, self.ny as i64 - 1) as usize;
        r * self.nx + c
    }

    /// Cells overlapping the box `x ± r`, `y ± r`, as (cell index) ranges per row.
    fn around(&self, x: f64, y: f64, r: f64) -> impl Iterator<Item = usize> + '_ {
        let c0 = self.col(x - r).max(0);
        let c1 = self.col(x + r).min(self.nx as i64 - 1);
        let r0 = self.row(y - r).max(0);
        let r1 = self.row(y + r).min(self.ny as i64 - 1);
        (r0..=r1).flat_map(move |row| (c0..=c1).map(move |c| row as usize * self.nx + c as usize))
    }

    fn members(&self, cell: usize) -> &[u32] {
        &self.idx[self.start[cell] as usize..self.start[cell + 1] as usize]
    }
}

/// Tree tops found by the local maximum filter on points: indices of the
/// points at or above `hmin` with no higher point in their window, in
/// order of x, then y (then index). `h` is the height of each point; points
/// with a non-finite coordinate or height take no part.
pub fn local_maxima_points(xy: &[[f64; 2]], h: &[f64], window: &Window, hmin: f64, shape: Shape) -> Result<Vec<usize>> {
    window.check()?;
    if xy.len() != h.len() {
        return Err(Error::invalid(format!("{} points but {} heights", xy.len(), h.len())));
    }
    if let Window::PerSite(v) = window {
        if v.len() != h.len() {
            return Err(Error::invalid(format!("{} window sizes for {} points", v.len(), h.len())));
        }
    }
    let sites: Vec<usize> = (0..h.len()).filter(|&i| h[i].is_finite() && xy[i][0].is_finite() && xy[i][1].is_finite()).collect();
    let cand: Vec<usize> = sites.iter().copied().filter(|&i| h[i] >= hmin).collect();
    let ws: Vec<f64> = cand.par_iter().map(|&i| site_window(window, i, h[i])).collect::<Result<_>>()?;
    let rmin = ws.iter().copied().fold(f64::INFINITY, f64::min) / 2.0;
    if cand.is_empty() {
        return Ok(Vec::new());
    }
    // Cells small enough that a cell lies within any window around a point in it.
    let grid = Buckets::new(xy, &sites, (rmin / std::f64::consts::SQRT_2).max(0.05));
    let mut cellmax = vec![f64::NEG_INFINITY; grid.nx * grid.ny];
    for &i in &sites {
        let c = grid.cell_of(xy[i][0], xy[i][1]);
        cellmax[c] = cellmax[c].max(h[i]);
    }
    let prefilter = grid.cell * std::f64::consts::SQRT_2 <= rmin;
    let keep: Vec<bool> = cand
        .par_iter()
        .zip(&ws)
        .map(|(&i, &w)| {
            let (x, y, z, r) = (xy[i][0], xy[i][1], h[i], w / 2.0);
            if prefilter && cellmax[grid.cell_of(x, y)] > z {
                return false;
            }
            for c in grid.around(x, y, r) {
                if cellmax[c] <= z {
                    continue;
                }
                for &j in grid.members(c) {
                    let j = j as usize;
                    if h[j] > z && shape.contains(xy[j][0] - x, xy[j][1] - y, r) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();
    let mut sel: Vec<(usize, f64)> = cand.iter().zip(&ws).zip(&keep).filter(|(_, k)| **k).map(|((&i, &w), _)| (i, w / 2.0)).collect();
    sel.sort_by(|a, b| xy[a.0][0].total_cmp(&xy[b.0][0]).then(xy[a.0][1].total_cmp(&xy[b.0][1])).then(a.0.cmp(&b.0)));
    let mut accepted: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tops = Vec::new();
    for (i, r) in sel {
        let (x, y, z) = (xy[i][0], xy[i][1], h[i]);
        let tie = grid.around(x, y, r).any(|c| accepted.get(&c).is_some_and(|v| v.iter().any(|&j| h[j] == z && shape.contains(xy[j][0] - x, xy[j][1] - y, r))));
        if !tie {
            accepted.entry(grid.cell_of(x, y)).or_default().push(i);
            tops.push(i);
        }
    }
    Ok(tops)
}

// ------------------------------------------------------------------ crowns on a CHM

/// A raster of seed ids (0 for none) with each top at its cell; of several
/// tops in one cell the highest (then the first) is kept. `tops` are
/// `(x, y, height)`; seed `k` is `tops[k]`, id `k + 1`.
pub fn seed_raster(chm: &Raster, tops: &[[f64; 3]]) -> Vec<u32> {
    let mut seeds = vec![0u32; chm.data.len()];
    for (k, t) in tops.iter().enumerate() {
        let (r, c) = chm.cell_index(t[0], t[1]);
        if !chm.in_bounds(r, c) {
            continue;
        }
        let i = r as usize * chm.ncols + c as usize;
        if seeds[i] == 0 || t[2] > tops[seeds[i] as usize - 1][2] {
            seeds[i] = k as u32 + 1;
        }
    }
    seeds
}

#[derive(PartialEq)]
struct Flood {
    h: f64,
    age: u64,
    cell: usize,
    label: u32,
}

impl Eq for Flood {}

impl Ord for Flood {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        // Highest first, then first pushed.
        self.h.total_cmp(&o.h).then(o.age.cmp(&self.age))
    }
}

impl PartialOrd for Flood {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}

/// Marker-controlled watershed of a CHM: every cell higher than `th_tree`
/// reachable from a seed through such cells gets the id of the seed whose
/// flood reaches it first, flooding from the highest cells down (Meyer's
/// algorithm on the inverted CHM, 8-connected, ties in height taken in the
/// order they were reached). Seeds on cells not higher than `th_tree` are
/// ignored. Returns an id per cell, 0 for none.
pub fn watershed(chm: &Raster, seeds: &[u32], th_tree: f64) -> Result<Vec<u32>> {
    if seeds.len() != chm.data.len() {
        return Err(Error::invalid(format!("{} seed cells for a raster of {} cells", seeds.len(), chm.data.len())));
    }
    let (nr, nc) = (chm.nrows as i64, chm.ncols as i64);
    let ok = |i: usize| chm.data[i] > th_tree;
    let mut label = vec![0u32; seeds.len()];
    let mut heap = BinaryHeap::new();
    let mut age = 0u64;
    // Seeds in order of x, then y.
    let mut order: Vec<usize> = (0..seeds.len()).filter(|&i| seeds[i] != 0 && ok(i)).collect();
    order.sort_by_key(|&i| (i % chm.ncols, i / chm.ncols));
    for i in order {
        label[i] = seeds[i];
        heap.push(Flood { h: chm.data[i], age, cell: i, label: seeds[i] });
        age += 1;
    }
    const NB: [(i64, i64); 8] = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)];
    while let Some(f) = heap.pop() {
        if label[f.cell] == 0 {
            label[f.cell] = f.label;
        } else if label[f.cell] != f.label {
            continue;
        } else if seeds[f.cell] != f.label {
            // Already reached by the same crown.
            continue;
        }
        let (row, col) = ((f.cell as i64) / nc, (f.cell as i64) % nc);
        for (dr, dc) in NB {
            let (rr, cc) = (row + dr, col + dc);
            if rr < 0 || rr >= nr || cc < 0 || cc >= nc {
                continue;
            }
            let j = (rr * nc + cc) as usize;
            if label[j] == 0 && ok(j) {
                heap.push(Flood { h: chm.data[j], age, cell: j, label: f.label });
                age += 1;
            }
        }
    }
    Ok(label)
}

/// Settings of [`dalponte2016`], with lidR's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dalponte {
    /// Cells not higher than this are never added to a crown (m).
    pub th_tree: f64,
    /// A cell joins only if higher than `th_seed` times the seed's height.
    pub th_seed: f64,
    /// ... and higher than `th_cr` times the crown's current mean height.
    pub th_cr: f64,
    /// ... and fewer than `max_cr` cells from the seed in x and in y.
    pub max_cr: f64,
}

impl Default for Dalponte {
    fn default() -> Self {
        Dalponte { th_tree: 2.0, th_seed: 0.45, th_cr: 0.55, max_cr: 10.0 }
    }
}

impl Dalponte {
    fn check(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.th_seed) || !(0.0..=1.0).contains(&self.th_cr) {
            return Err(Error::invalid(format!("th_seed and th_cr must be between 0 and 1, got {} and {}", self.th_seed, self.th_cr)));
        }
        if !self.th_tree.is_finite() || !(self.max_cr.is_finite() && self.max_cr > 0.0) {
            return Err(Error::invalid(format!("th_tree must be a number and max_cr a positive number of cells, got {} and {}", self.th_tree, self.max_cr)));
        }
        Ok(())
    }
}

/// Region growing of Dalponte and Coomes (2016), as lidR's
/// `C_dalponte2016` does it. The image is scanned by columns from west to
/// east and, within a column, by rows from south to north (lidR's matrix
/// order), skipping the outermost cells; each crown cell adds a 4-neighbour
/// that is not yet in a crown (at the start of the sweep), is higher than
/// `th_tree`, than `th_seed` times the seed's CHM value and than `th_cr`
/// times the crown's mean height, is at most 5 % above the seed, and lies
/// fewer than `max_cr` cells from the seed in x and in y. Additions made in
/// a sweep take effect for the next one (a cell claimed twice in a sweep
/// goes to the later claim); the mean heights are updated as cells are
/// added, counting a cell each time it is claimed, as lidR does. Sweeps
/// repeat until nothing grows. NaN cells are `-inf`. Returns an id per
/// cell, 0 for none.
pub fn dalponte2016(chm: &Raster, seeds: &[u32], p: &Dalponte) -> Result<Vec<u32>> {
    p.check()?;
    if seeds.len() != chm.data.len() {
        return Err(Error::invalid(format!("{} seed cells for a raster of {} cells", seeds.len(), chm.data.len())));
    }
    let (nr, nc) = (chm.nrows, chm.ncols);
    let img = |col: usize, row: usize| {
        let v = chm.data[row * nc + col];
        if v.is_nan() {
            f64::NEG_INFINITY
        } else {
            v
        }
    };
    let n_ids = seeds.iter().copied().max().unwrap_or(0) as usize;
    let mut seed_at = vec![(0usize, 0usize); n_ids + 1];
    let mut sum = vec![0.0; n_ids + 1];
    let mut npix = vec![0.0; n_ids + 1];
    // lidR records the last cell of each id in its (column-major) scan of the seed matrix.
    for col in 0..nc {
        for row in 0..nr {
            let id = seeds[row * nc + col] as usize;
            if id != 0 {
                seed_at[id] = (col, row);
                sum[id] = img(col, row);
                npix[id] = 1.0;
            }
        }
    }
    let mut region = seeds.to_vec();
    let mut temp = seeds.to_vec();
    let mut grown = true;
    while grown {
        grown = false;
        for col in 1..nc.saturating_sub(1) {
            for row in 1..nr.saturating_sub(1) {
                let id = region[row * nc + col];
                if id == 0 {
                    continue;
                }
                let idu = id as usize;
                let (sc, sr) = seed_at[idu];
                let hseed = img(sc, sr);
                let mh = sum[idu] / npix[idu];
                for (c2, r2) in [(col - 1, row), (col, row - 1), (col, row + 1), (col + 1, row)] {
                    let z = img(c2, r2);
                    if z > p.th_tree {
                        let expand = z > hseed * p.th_seed && z > mh * p.th_cr && z <= hseed + hseed * 0.05 && ((sc as f64) - c2 as f64).abs() < p.max_cr && ((sr as f64) - r2 as f64).abs() < p.max_cr && region[r2 * nc + c2] == 0;
                        if expand {
                            temp[r2 * nc + c2] = id;
                            npix[idu] += 1.0;
                            sum[idu] += z;
                            grown = true;
                        }
                    }
                }
            }
        }
        region.copy_from_slice(&temp);
    }
    Ok(region)
}

// ------------------------------------------------------------------ Li et al. 2012

/// Settings of [`li2012`], with lidR's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Li2012 {
    /// Spacing threshold (m) for points up to `zu` high.
    pub dt1: f64,
    /// Spacing threshold (m) for points higher than `zu`.
    pub dt2: f64,
    /// Diameter (m) of the local maximum window (lidR passes `R` as the
    /// window size of its local maximum filter); 0 makes every point a
    /// local maximum.
    pub r: f64,
    /// Height (m) above which `dt2` applies.
    pub zu: f64,
    /// The segmentation stops when the highest point left is below this (m).
    pub hmin: f64,
    /// Points farther than this (m) from a tree's top are not considered
    /// for that tree; it only saves time if larger than any crown.
    pub speed_up: f64,
}

impl Default for Li2012 {
    fn default() -> Self {
        Li2012 { dt1: 1.5, dt2: 2.0, r: 2.0, zu: 15.0, hmin: 2.0, speed_up: 10.0 }
    }
}

impl Li2012 {
    fn check(&self) -> Result<()> {
        let pos = |v: f64| v.is_finite() && v > 0.0;
        if !(pos(self.dt1) && pos(self.dt2) && pos(self.zu) && pos(self.hmin) && pos(self.speed_up) && self.r.is_finite() && self.r >= 0.0) {
            return Err(Error::invalid(format!("li2012 needs positive dt1, dt2, Zu, hmin and speed_up and R >= 0, got dt1 {}, dt2 {}, R {}, Zu {}, hmin {}, speed_up {}", self.dt1, self.dt2, self.r, self.zu, self.hmin, self.speed_up)));
        }
        Ok(())
    }
}

/// P and N of one tree: points bucketed around its top for nearest-distance queries.
struct NearSet {
    x0: f64,
    y0: f64,
    cell: f64,
    n: usize,
    cells: Vec<Vec<[f64; 2]>>,
    used: Vec<usize>,
    count: usize,
}

impl NearSet {
    fn new(cell: f64, n: usize) -> NearSet {
        NearSet { x0: 0.0, y0: 0.0, cell, n, cells: vec![Vec::new(); n * n], used: Vec::new(), count: 0 }
    }

    fn reset(&mut self, x0: f64, y0: f64) {
        for &c in &self.used {
            self.cells[c].clear();
        }
        self.used.clear();
        self.count = 0;
        self.x0 = x0;
        self.y0 = y0;
    }

    fn cell_of(&self, x: f64, y: f64) -> (i64, i64) {
        let f = |v: f64, o: f64| (((v - o) / self.cell).floor() as i64).clamp(0, self.n as i64 - 1);
        (f(x, self.x0), f(y, self.y0))
    }

    fn push(&mut self, p: [f64; 2]) {
        let (c, r) = self.cell_of(p[0], p[1]);
        let k = r as usize * self.n + c as usize;
        if self.cells[k].is_empty() {
            self.used.push(k);
        }
        self.cells[k].push(p);
        self.count += 1;
    }

    /// Smallest squared distance from `(x, y)` to the set (infinite if empty).
    fn nearest_sq(&self, x: f64, y: f64) -> f64 {
        if self.count == 0 {
            return f64::INFINITY;
        }
        let (c0, r0) = self.cell_of(x, y);
        let mut best = f64::INFINITY;
        let n = self.n as i64;
        for ring in 0..n {
            for r in (r0 - ring)..=(r0 + ring) {
                if r < 0 || r >= n {
                    continue;
                }
                let edge = r == r0 - ring || r == r0 + ring;
                let mut c = c0 - ring;
                while c <= c0 + ring {
                    if c >= 0 && c < n {
                        for q in &self.cells[r as usize * self.n + c as usize] {
                            let (dx, dy) = (q[0] - x, q[1] - y);
                            let d = dx * dx + dy * dy;
                            if d < best {
                                best = d;
                            }
                        }
                    }
                    c += if edge || ring == 0 { 1 } else { 2 * ring };
                }
            }
            let reach = ring as f64 * self.cell;
            if best <= reach * reach {
                break;
            }
        }
        best
    }
}

/// Point-based segmentation of Li et al. (2012), as lidR's
/// `LAS::segment_trees` does it. Points are taken from the highest down
/// (ties by x, then y, then index); the highest point left starts tree `k`
/// (its set P) with an empty set N (lidR's dummy point, which no point
/// within `speed_up` of the top can be nearer than). Every point left
/// within `speed_up` of that top, from the highest down, joins P or N by
/// its smallest distances `d1` to P and `d2` to N (in x, y): a local
/// maximum (within a window of diameter `r`, [`local_maxima_points`] with
/// `hmin` 0) joins N if `d1 > dt` or `d2 < d1 < dt`, P otherwise, with `dt`
/// `dt2` above `zu` and `dt1` below; any other point joins P if `d1 <= d2`.
/// P becomes tree `k` and N is left for the following trees. The loop stops
/// when the highest point left is lower than `hmin`. Returns the tree of
/// each point (1, 2, ... in the order found), 0 for none; the first point
/// of each tree in that order is its top.
pub fn li2012(xy: &[[f64; 2]], h: &[f64], p: &Li2012) -> Result<Vec<u32>> {
    p.check()?;
    if xy.len() != h.len() {
        return Err(Error::invalid(format!("{} points but {} heights", xy.len(), h.len())));
    }
    let n = h.len();
    let valid: Vec<usize> = (0..n).filter(|&i| h[i].is_finite() && xy[i][0].is_finite() && xy[i][1].is_finite()).collect();
    let mut is_lm = vec![false; n];
    if p.r > 0.0 {
        for i in local_maxima_points(xy, h, &Window::Fixed(p.r), 0.0, Shape::Circular)? {
            is_lm[i] = true;
        }
    } else {
        is_lm.iter_mut().for_each(|v| *v = true);
    }
    let mut order = valid.clone();
    order.sort_by(|&a, &b| h[b].total_cmp(&h[a]).then(xy[a][0].total_cmp(&xy[b][0])).then(xy[a][1].total_cmp(&xy[b][1])).then(a.cmp(&b)));
    let mut rank = vec![u32::MAX; n];
    for (k, &i) in order.iter().enumerate() {
        rank[i] = k as u32;
    }
    let grid = Buckets::new(xy, &valid, (p.speed_up / 4.0).max(0.25));
    let mut alive = vec![false; n];
    for &i in &valid {
        alive[i] = true;
    }
    let mut label = vec![0u32; n];
    let (radius, dt1, dt2) = (p.speed_up * p.speed_up, p.dt1 * p.dt1, p.dt2 * p.dt2);
    let cell = (p.speed_up / 20.0).max(0.25);
    let side = ((2.0 * p.speed_up) / cell).ceil() as usize + 2;
    let (mut pset, mut nset) = (NearSet::new(cell, side), NearSet::new(cell, side));
    let mut next = 0usize;
    let mut k = 0u32;
    let mut near: Vec<usize> = Vec::new();
    loop {
        while next < order.len() && !alive[order[next]] {
            next += 1;
        }
        if next >= order.len() {
            break;
        }
        let u = order[next];
        if h[u] < p.hmin {
            break;
        }
        k += 1;
        let (ux, uy) = (xy[u][0], xy[u][1]);
        alive[u] = false;
        label[u] = k;
        pset.reset(ux - p.speed_up - cell, uy - p.speed_up - cell);
        nset.reset(ux - p.speed_up - cell, uy - p.speed_up - cell);
        pset.push(xy[u]);
        near.clear();
        for c in grid.around(ux, uy, p.speed_up) {
            for &j in grid.members(c) {
                let j = j as usize;
                if alive[j] {
                    let (dx, dy) = (xy[j][0] - ux, xy[j][1] - uy);
                    if dx * dx + dy * dy <= radius {
                        near.push(j);
                    }
                }
            }
        }
        near.sort_unstable_by_key(|&j| rank[j]);
        for &v in &near {
            let (x, y) = (xy[v][0], xy[v][1]);
            let d1 = pset.nearest_sq(x, y);
            let d2 = nset.nearest_sq(x, y);
            let dt = if h[v] > p.zu { dt2 } else { dt1 };
            let to_p = if is_lm[v] { !(d1 > dt || (d1 < dt && d1 > d2)) } else { d1 <= d2 };
            if to_p {
                pset.push(xy[v]);
                label[v] = k;
                alive[v] = false;
            } else {
                nset.push(xy[v]);
            }
        }
    }
    Ok(label)
}

// ------------------------------------------------------------------ hulls

/// Outline of a crown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hull {
    /// Convex hull.
    Convex,
    /// Characteristic shape (chi-shape) of Duckham et al. (2008): from the
    /// Delaunay triangulation, boundary edges longer than this (m) are
    /// removed, longest first, as long as the outline stays a simple
    /// polygon.
    Concave(f64),
}

/// Counter-clockwise outline of 2-D points (without repeating the first
/// vertex); fewer than 3 distinct points, or collinear ones, give their
/// convex hull.
pub fn hull(xy: &[[f64; 2]], how: Hull) -> Vec<[f64; 2]> {
    match how {
        Hull::Convex => convex_hull(xy),
        Hull::Concave(l) => chi_shape(xy, l),
    }
}

/// The chi-shape of Duckham et al. (2008) with edge length threshold `l`.
pub fn chi_shape(xy: &[[f64; 2]], l: f64) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = xy.iter().copied().filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 4 {
        return convex_hull(&pts);
    }
    let tri: DelaunayTriangulation<Point2<f64>> = match DelaunayTriangulation::bulk_load_stable(pts.iter().map(|p| Point2::new(p[0], p[1])).collect()) {
        Ok(t) => t,
        Err(_) => return convex_hull(&pts),
    };
    let pos: Vec<[f64; 2]> = tri.vertices().map(|v| [v.position().x, v.position().y]).collect();
    let tris: Vec<[usize; 3]> = tri
        .inner_faces()
        .map(|f| {
            let v = f.vertices();
            [v[0].fix().index(), v[1].fix().index(), v[2].fix().index()]
        })
        .collect();
    if tris.is_empty() {
        return convex_hull(&pts);
    }
    let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (t, v) in tris.iter().enumerate() {
        for e in 0..3 {
            edge_tris.entry(key(v[e], v[(e + 1) % 3])).or_default().push(t);
        }
    }
    let mut alive_tri = vec![true; tris.len()];
    let mut on_boundary = vec![false; pos.len()];
    let len = |e: (usize, usize)| ((pos[e.0][0] - pos[e.1][0]).powi(2) + (pos[e.0][1] - pos[e.1][1]).powi(2)).sqrt();
    #[derive(PartialEq)]
    struct E(f64, (usize, usize));
    impl Eq for E {}
    impl Ord for E {
        fn cmp(&self, o: &Self) -> CmpOrdering {
            self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
        }
    }
    impl PartialOrd for E {
        fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
            Some(self.cmp(o))
        }
    }
    let mut heap = BinaryHeap::new();
    let mut boundary: Vec<(usize, usize)> = edge_tris.iter().filter(|(_, t)| t.len() == 1).map(|(e, _)| *e).collect();
    boundary.sort_unstable();
    for e in boundary {
        on_boundary[e.0] = true;
        on_boundary[e.1] = true;
        heap.push(E(len(e), e));
    }
    let alive_of = |e: &(usize, usize), alive: &[bool]| -> Vec<usize> { edge_tris[e].iter().copied().filter(|&t| alive[t]).collect() };
    while let Some(E(d, e)) = heap.pop() {
        if d <= l {
            break;
        }
        let ts = alive_of(&e, &alive_tri);
        if ts.len() != 1 {
            continue;
        }
        let t = ts[0];
        let c = tris[t].iter().copied().find(|&v| v != e.0 && v != e.1).expect("a triangle has a third vertex");
        if on_boundary[c] {
            continue;
        }
        alive_tri[t] = false;
        on_boundary[c] = true;
        for f in [key(e.0, c), key(e.1, c)] {
            heap.push(E(len(f), f));
        }
    }
    // Trace the outline counter-clockwise: each boundary edge in the
    // orientation of its (counter-clockwise) triangle.
    let mut next: HashMap<usize, usize> = HashMap::new();
    for (t, v) in tris.iter().enumerate() {
        if !alive_tri[t] {
            continue;
        }
        let ccw = {
            let (a, b, c) = (pos[v[0]], pos[v[1]], pos[v[2]]);
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0
        };
        for e in 0..3 {
            let (a, b) = if ccw { (v[e], v[(e + 1) % 3]) } else { (v[(e + 1) % 3], v[e]) };
            if alive_of(&key(a, b), &alive_tri).len() == 1 {
                next.insert(a, b);
            }
        }
    }
    let Some(&start) = next.keys().min() else { return convex_hull(&pts) };
    let mut out = vec![pos[start]];
    let mut cur = next[&start];
    while cur != start && out.len() <= next.len() {
        out.push(pos[cur]);
        cur = match next.get(&cur) {
            Some(&n) => n,
            None => break,
        };
    }
    out
}

// ------------------------------------------------------------------ one cloud

/// How crowns are delineated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    /// Tree tops only.
    Tops,
    /// [`watershed`] of the CHM with `th_tree`.
    Watershed { th_tree: f64 },
    /// [`dalponte2016`] on the CHM.
    Dalponte(Dalponte),
    /// [`li2012`] on the points.
    Li2012(Li2012),
}

/// Where the tops that seed CHM crowns come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopsFrom {
    /// [`local_maxima_raster`] on the CHM.
    Chm,
    /// [`local_maxima_points`] on the points (seeding the cell of each).
    Points,
}

/// Settings of a segmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeParams {
    pub method: Method,
    /// CHM cell size (m).
    pub resolution: f64,
    pub window: Window,
    /// Lowest tree top (m).
    pub hmin: f64,
    pub shape: Shape,
    pub tops_from: TopsFrom,
    pub hull: Hull,
    /// Points lower than this (m) are not labelled and not part of any crown.
    pub min_point_height: f64,
    /// Smooth the CHM with a mean filter over `(2 smooth + 1)²` cells
    /// ([`Raster::smooth`]) before finding tops and growing crowns; 0 for none.
    pub smooth: usize,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams { method: Method::Dalponte(Dalponte::default()), resolution: 0.5, window: Window::Fixed(5.0), hmin: 2.0, shape: Shape::Circular, tops_from: TopsFrom::Chm, hull: Hull::Convex, min_point_height: 0.5, smooth: 0 }
    }
}

impl TreeParams {
    fn check(&self) -> Result<()> {
        self.window.check()?;
        if !(self.resolution.is_finite() && self.resolution > 0.0) {
            return Err(Error::invalid(format!("resolution must be a positive number, got {}", self.resolution)));
        }
        if !self.hmin.is_finite() || !self.min_point_height.is_finite() {
            return Err(Error::invalid("hmin and min_point_height must be numbers"));
        }
        if let Hull::Concave(l) = self.hull {
            if !(l.is_finite() && l > 0.0) {
                return Err(Error::invalid(format!("the concave hull's edge length must be a positive number of metres, got {l}")));
            }
        }
        match self.method {
            Method::Dalponte(d) => d.check(),
            Method::Li2012(l) => l.check(),
            Method::Watershed { th_tree } if !th_tree.is_finite() => Err(Error::invalid("th_tree must be a number")),
            _ => Ok(()),
        }
    }

    fn uses_chm(&self) -> bool {
        !matches!(self.method, Method::Li2012(_)) && !(matches!(self.method, Method::Tops) && self.tops_from == TopsFrom::Points)
    }
}

/// One tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// 1, 2, ... in order of x, then y, of the tops.
    pub id: u32,
    /// Top: a CHM cell centre and its value, or a point and its height.
    pub x: f64,
    pub y: f64,
    pub height: f64,
    /// Area (m²) of the crown outline; NaN without a crown (tops only, or
    /// fewer than 3 points).
    pub crown_area: f64,
    /// Points labelled with the tree.
    pub n_points: usize,
    /// Crown outline, counter-clockwise, from the tree's points.
    pub crown: Vec<[f64; 2]>,
}

/// A tree top as a key: its exact coordinates.
type TopKey = [u64; 3];

fn top_key(x: f64, y: f64, h: f64) -> TopKey {
    [x.to_bits(), y.to_bits(), h.to_bits()]
}

/// What a segmentation of one chunk gives.
struct ChunkTrees {
    /// Trees whose top the chunk owns (ids unset).
    trees: Vec<(TopKey, Tree)>,
    /// Every top found, owned or not.
    keys: Vec<TopKey>,
    /// Per point: 1 + index into `keys`, 0 for none.
    labels: Vec<u32>,
    /// Owned crowns that reach an inner edge of the buffered box.
    at_edge: usize,
}

/// Segment the points `xyz` with heights `h`. `chm_bounds` gives the CHM
/// grid (as [`Raster::from_points`] takes bounds); `owns_cell` says which
/// CHM-cell tops the chunk owns and `is_core` which points (for point
/// tops). `inner` flags the sides of `outer` (west, south, east, north)
/// that are buffer edges rather than the survey's edge.
#[allow(clippy::too_many_arguments)]
fn segment_points(xyz: &[Point], h: &[f64], is_core: &[bool], chm_bounds: (f64, f64, f64, f64), p: &TreeParams, owns_cell: &dyn Fn(f64, f64) -> bool, outer: [f64; 4], inner: [bool; 4]) -> Result<ChunkTrees> {
    let n = xyz.len();
    let xy: Vec<[f64; 2]> = xyz.iter().map(|q| [q[0], q[1]]).collect();
    let labelled = |i: usize| h[i].is_finite() && h[i] >= p.min_point_height;
    let mut keys: Vec<TopKey> = Vec::new();
    let mut tops: Vec<[f64; 3]> = Vec::new();
    let mut owned: Vec<bool> = Vec::new();
    let mut labels = vec![0u32; n];
    let mut crown_cells: Option<(Raster, Vec<u32>)> = None;
    match p.method {
        Method::Li2012(li) => {
            let idx: Vec<usize> = (0..n).filter(|&i| labelled(i)).collect();
            let sxy: Vec<[f64; 2]> = idx.iter().map(|&i| xy[i]).collect();
            let sh: Vec<f64> = idx.iter().map(|&i| h[i]).collect();
            let l = li2012(&sxy, &sh, &li)?;
            let ntrees = l.iter().copied().max().unwrap_or(0) as usize;
            let mut first = vec![usize::MAX; ntrees + 1];
            // The top of a tree is its highest point: the first in li2012's order.
            let mut order: Vec<usize> = (0..idx.len()).filter(|&k| l[k] != 0).collect();
            order.sort_by(|&a, &b| sh[b].total_cmp(&sh[a]).then(sxy[a][0].total_cmp(&sxy[b][0])).then(sxy[a][1].total_cmp(&sxy[b][1])).then(a.cmp(&b)));
            for k in order {
                let t = l[k] as usize;
                if first[t] == usize::MAX {
                    first[t] = k;
                }
            }
            for &k in first.iter().skip(1) {
                let i = idx[k];
                keys.push(top_key(xy[i][0], xy[i][1], h[i]));
                tops.push([xy[i][0], xy[i][1], h[i]]);
                owned.push(is_core[i]);
            }
            for (k, &i) in idx.iter().enumerate() {
                labels[i] = l[k];
            }
        }
        _ => {
            // The raw CHM gives the heights of CHM tops; the smoothed one the tops and crowns.
            let raw = if p.uses_chm() { Some(ground::make_chm(xyz, h, p.resolution, Some(chm_bounds), 0.0)?) } else { None };
            let chm = raw.as_ref().map(|r| if p.smooth > 0 { r.smooth(p.smooth) } else { r.clone() });
            match p.tops_from {
                TopsFrom::Chm => {
                    let (chm, raw) = (chm.as_ref().expect("a CHM for CHM tops"), raw.as_ref().expect("a CHM"));
                    for c in local_maxima_raster(chm, &p.window, p.hmin, p.shape)? {
                        let (x, y) = chm.cell_center(c / chm.ncols, c % chm.ncols);
                        let z = raw.data[c];
                        keys.push(top_key(x, y, z));
                        tops.push([x, y, z]);
                        owned.push(owns_cell(x, y));
                    }
                }
                TopsFrom::Points => {
                    for i in local_maxima_points(&xy, h, &p.window, p.hmin, p.shape)? {
                        keys.push(top_key(xy[i][0], xy[i][1], h[i]));
                        tops.push([xy[i][0], xy[i][1], h[i]]);
                        owned.push(is_core[i]);
                    }
                }
            }
            if let Some(chm) = chm {
                let seeds = seed_raster(&chm, &tops);
                let crowns = match p.method {
                    Method::Watershed { th_tree } => Some(watershed(&chm, &seeds, th_tree)?),
                    Method::Dalponte(d) => Some(dalponte2016(&chm, &seeds, &d)?),
                    _ => None,
                };
                if let Some(crowns) = crowns {
                    for i in 0..n {
                        if labelled(i) {
                            let (r, c) = chm.cell_index(xy[i][0], xy[i][1]);
                            if chm.in_bounds(r, c) {
                                labels[i] = crowns[r as usize * chm.ncols + c as usize];
                            }
                        }
                    }
                    crown_cells = Some((chm, crowns));
                }
            }
        }
    }
    // Points of each owned tree.
    let crowns_made = p.method != Method::Tops;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); tops.len()];
    if crowns_made {
        for (i, &l) in labels.iter().enumerate() {
            let l = l as usize;
            if l != 0 && owned[l - 1] {
                members[l - 1].push(i);
            }
        }
    }
    let mut at_edge = 0;
    let near = |x: f64, y: f64, tol: f64| (inner[0] && x - outer[0] <= tol) || (inner[1] && y - outer[1] <= tol) || (inner[2] && outer[2] - x <= tol) || (inner[3] && outer[3] - y <= tol);
    if let Some((chm, crowns)) = &crown_cells {
        let tol = chm.resolution;
        let mut edge = vec![false; tops.len()];
        for (c, &l) in crowns.iter().enumerate() {
            if l != 0 && owned[l as usize - 1] && !edge[l as usize - 1] {
                let (x, y) = chm.cell_center(c / chm.ncols, c % chm.ncols);
                edge[l as usize - 1] = near(x, y, tol);
            }
        }
        at_edge = edge.iter().filter(|&&e| e).count();
    } else if matches!(p.method, Method::Li2012(_)) {
        at_edge = members.iter().filter(|m| m.iter().any(|&i| near(xy[i][0], xy[i][1], 1.0))).count();
    }
    let trees: Vec<(TopKey, Tree)> = (0..tops.len())
        .into_par_iter()
        .filter(|&k| owned[k])
        .map(|k| {
            let t = tops[k];
            let (crown, area) = if crowns_made {
                let pts: Vec<[f64; 2]> = members[k].iter().map(|&i| xy[i]).collect();
                let c = hull(&pts, p.hull);
                let a = if c.len() >= 3 { polygon_area(&c) } else { f64::NAN };
                (c, a)
            } else {
                (Vec::new(), f64::NAN)
            };
            (keys[k], Tree { id: 0, x: t[0], y: t[1], height: t[2], crown_area: area, n_points: members[k].len(), crown })
        })
        .collect();
    Ok(ChunkTrees { trees, keys, labels, at_edge })
}

fn order_trees(trees: &mut [(TopKey, Tree)]) {
    trees.sort_by(|a, b| a.1.x.total_cmp(&b.1.x).then(a.1.y.total_cmp(&b.1.y)).then(a.1.height.total_cmp(&b.1.height)));
    for (k, t) in trees.iter_mut().enumerate() {
        t.1.id = k as u32 + 1;
    }
}

/// Trees of one cloud: `h` is the height above ground of each point (z of
/// a normalised cloud). The CHM grid is aligned with multiples of the
/// resolution, as a catalogue's grid is, and has two empty cells around
/// the cloud.
/// Returns the trees (ids 1, 2, ... in order of x, then y, of the tops)
/// and each point's tree id (0 for none).
pub fn segment_cloud(xyz: &[Point], h: &[f64], p: &TreeParams) -> Result<(Vec<Tree>, Vec<u32>)> {
    p.check()?;
    if xyz.len() != h.len() {
        return Err(Error::invalid(format!("{} points but {} heights", xyz.len(), h.len())));
    }
    let finite: Vec<&Point> = xyz.iter().filter(|q| q[0].is_finite() && q[1].is_finite()).collect();
    if finite.is_empty() {
        return Ok((Vec::new(), vec![0; xyz.len()]));
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for q in finite {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let res = p.resolution;
    // Two empty cells around the cloud, as a chunk's buffer gives at the
    // survey's edge, so that crowns can reach the outermost points.
    let b = ((lo[0] / res).floor() * res - 2.0 * res, (lo[1] / res).floor() * res - 2.0 * res, hi[0] + 2.0 * res, hi[1] + 2.0 * res);
    crate::limits::check_cells((((hi[0] - b.0) / res) as u128 + 1) * (((hi[1] - b.1) / res) as u128 + 1), 17, "the CHM", "a coarser resolution")?;
    let core = vec![true; xyz.len()];
    let c = segment_points(xyz, h, &core, b, p, &|_, _| true, [lo[0], lo[1], hi[0], hi[1]], [false; 4])?;
    let mut trees = c.trees;
    order_trees(&mut trees);
    let id: HashMap<TopKey, u32> = trees.iter().map(|(k, t)| (*k, t.id)).collect();
    let map: Vec<u32> = c.keys.iter().map(|k| id[k]).collect();
    let labels = c.labels.iter().map(|&l| if l == 0 { 0 } else { map[l as usize - 1] }).collect();
    Ok((trees.into_iter().map(|t| t.1).collect(), labels))
}

// ------------------------------------------------------------------ a catalogue

/// The result of [`catalog_trees`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CatalogTrees {
    pub trees: Vec<Tree>,
    /// Tiles written with the tree ids, in chunk order.
    pub written: Vec<PathBuf>,
    /// Crowns that reached the inner edge of their chunk's buffer: a sign
    /// that the buffer is too narrow for them to be complete.
    pub at_edge: usize,
    /// Points written without an id because the tree their chunk gave them
    /// was owned by no chunk (a crown whose top is found differently near
    /// a buffer's edge; 0 with an adequate buffer).
    pub unmatched_points: u64,
}

fn dist_to_box(b: &[f64; 4], x: f64, y: f64) -> f64 {
    if crate::als::in_core(b, x, y) {
        return 0.0;
    }
    let dx = (b[0] - x).max(0.0).max(x - b[2]);
    let dy = (b[1] - y).max(0.0).max(y - b[3]);
    (dx * dx + dy * dy).sqrt().max(f64::MIN_POSITIVE)
}

/// Is the cell centre `(x, y)` the business of chunk `k`: is its core the
/// nearest (0 inside the half-open core), ties to the lower index, as
/// [`crate::als::mosaic`] assigns cells?
fn owns(chunks: &[Chunk], k: usize, x: f64, y: f64) -> bool {
    let d = dist_to_box(&chunks[k].core, x, y);
    chunks.iter().enumerate().all(|(j, c)| {
        let e = dist_to_box(&c.core, x, y);
        e > d || (e == d && j >= k)
    })
}

/// Find the trees of a whole catalogue and, with `out_dir`, write every
/// tile with each point's tree id in the attribute `attribute` (0 for no
/// tree). Each tree is segmented in the chunk that owns its top (see the
/// module notes) with that chunk's buffer, so a crown that crosses a tile
/// edge is found once and whole if the buffer is wide enough. Ids are 1,
/// 2, ... in order of x, then y, of the tops, over the whole catalogue.
/// [`Heights::Auto`] makes one DTM of the whole catalogue first
/// ([`crate::als_ops::dtm`] with [`crate::als_ops::DtmMethod::Lowest`]), so
/// that every chunk gives a point the same height.
#[allow(clippy::too_many_arguments)]
pub fn catalog_trees(cat: &Catalog, p: &TreeParams, heights: &Heights, out_dir: Option<&Path>, attribute: &str, format: Option<&str>, opts: &RunOptions) -> Result<CatalogTrees> {
    p.check()?;
    heights.check()?;
    if matches!(p.window, Window::PerSite(_)) {
        return Err(Error::invalid("a catalogue needs a fixed, linear or tabulated window, not one value per site"));
    }
    if out_dir.is_some() && p.method == Method::Tops {
        return Err(Error::invalid("labelled tiles need crowns; choose a segmentation method"));
    }
    if attribute.is_empty() {
        return Err(Error::invalid("the attribute name is empty"));
    }
    let grid = catalog_grid(cat, p.resolution)?;
    // A point must have the same height in every chunk that reads it, or a
    // tree seen from two chunks could differ: so "auto" heights come from
    // one DTM of the whole catalogue rather than one per chunk.
    let whole_dtm;
    let heights = match heights {
        Heights::Auto { resolution } => {
            whole_dtm = Heights::Dtm(crate::als_ops::dtm(cat, *resolution, &crate::als_ops::DtmMethod::Lowest, opts)?);
            &whole_dtm
        }
        h => h,
    };
    let chunks = crate::als::plan(cat, opts.layout, opts.buffer)?;
    let w = workers_for(&chunks, opts.workers, BYTES_PER_POINT)?;
    let survey = cat.xy_bounds().expect("a catalogue with tiles");
    let keep_labels = out_dir.is_some();
    let parts = run(cat, &chunks, w, "segmenting trees", |chunk, data| {
        let Some(h) = chunk_heights(cat, chunk, &data.cloud, heights)? else { return Ok(None) };
        let is_core: Vec<bool> = data.buffer.iter().map(|&b| !b).collect();
        let b = chunk_bounds(&grid, &chunk.outer);
        let o = chunk.outer;
        let inner = [o[0] > survey[0], o[1] > survey[1], o[2] < survey[2], o[3] < survey[3]];
        let k = chunk.index;
        let mut c = segment_points(&data.cloud.xyz, &h, &is_core, b, p, &|x, y| owns(&chunks, k, x, y), o, inner)?;
        c.labels = if keep_labels { (0..c.labels.len()).filter(|&i| is_core[i]).map(|i| c.labels[i]).collect() } else { Vec::new() };
        Ok(Some(c))
    })?;
    let parts: Vec<Option<ChunkTrees>> = parts.into_iter().map(|p| p.flatten()).collect();
    let mut trees: Vec<(TopKey, Tree)> = Vec::new();
    let mut at_edge = 0;
    for c in parts.iter().flatten() {
        trees.extend(c.trees.iter().cloned());
        at_edge += c.at_edge;
    }
    order_trees(&mut trees);
    let ids: HashMap<TopKey, u32> = trees.iter().map(|(k, t)| (*k, t.id)).collect();
    let mut written = Vec::new();
    let mut unmatched = 0u64;
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;
        let ext = |c: &Chunk| -> Result<String> {
            if let Some(f) = format {
                let f = f.trim_start_matches('.').to_ascii_lowercase();
                if f != "las" && f != "laz" {
                    return Err(Error::invalid(format!("format must be 'las' or 'laz', got {f:?}")));
                }
                return Ok(f);
            }
            let t = &cat.tiles[c.own.unwrap_or(c.files[0])];
            Ok(t.path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).filter(|e| e == "las" || e == "laz").unwrap_or_else(|| "laz".to_string()))
        };
        for c in &chunks {
            output_path(cat, dir, &c.name, &ext(c)?)?;
        }
        // Read only the cores again, in the same order, to write them with their ids.
        let cores: Vec<Chunk> = chunks.iter().map(|c| Chunk { outer: c.core, ..c.clone() }).collect();
        let maps: Vec<Option<Vec<u32>>> = parts.iter().map(|c| c.as_ref().map(|c| c.keys.iter().map(|k| ids.get(k).copied().unwrap_or(0)).collect())).collect();
        let out = run(cat, &cores, w, "writing labelled tiles", |chunk, data| {
            let k = chunk.index;
            let core: Vec<usize> = data.core_indices();
            let mut cloud = data.cloud.take(&core);
            let (labels, map) = match (&parts[k], &maps[k]) {
                (Some(c), Some(m)) => (&c.labels[..], &m[..]),
                _ => (&[][..], &[][..]),
            };
            if !labels.is_empty() && labels.len() != cloud.len() {
                return Err(Error::invalid(format!("chunk {} read {} points the second time and {} the first", chunk.name, cloud.len(), labels.len())));
            }
            let mut miss = 0u64;
            let id: Vec<i32> = (0..cloud.len())
                .map(|i| {
                    let l = labels.get(i).copied().unwrap_or(0);
                    if l == 0 {
                        return 0;
                    }
                    let v = map[l as usize - 1];
                    if v == 0 {
                        miss += 1;
                    }
                    v as i32
                })
                .collect();
            cloud.attrs.insert(attribute.to_string(), Attr::I32(id));
            let path = output_path(cat, dir, &chunk.name, &ext(chunk)?)?;
            write_like(&cloud, &path, &cat.tiles[chunk.own.unwrap_or(chunk.files[0])])?;
            Ok((path, miss))
        })?;
        for (path, miss) in out.into_iter().flatten() {
            written.push(path);
            unmatched += miss;
        }
    }
    Ok(CatalogTrees { trees: trees.into_iter().map(|t| t.1).collect(), written, at_edge, unmatched_points: unmatched })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(rows: &[&[f64]]) -> Raster {
        // rows given north first; row 0 of a Raster is the south.
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut r = Raster::filled(nrows, ncols, 0.0, 0.0, 1.0, 0.0);
        for (k, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                r.set(nrows - 1 - k, c, v);
            }
        }
        r
    }

    /// Two cones of height 10 and 8, 12 m apart, on a 0.5 m grid.
    fn cones() -> Raster {
        let mut r = Raster::filled(40, 60, 0.0, 0.0, 0.5, 0.0);
        for row in 0..40 {
            for col in 0..60 {
                let (x, y) = r.cell_center(row, col);
                let a = 10.0 - 1.2 * ((x - 10.25).hypot(y - 10.25));
                let b = 8.0 - 1.2 * ((x - 22.25).hypot(y - 10.25));
                r.set(row, col, a.max(b).max(0.0));
            }
        }
        r
    }

    #[test]
    fn local_maxima_on_a_raster() {
        let r = cones();
        let tops = local_maxima_raster(&r, &Window::Fixed(3.0), 2.0, Shape::Circular).unwrap();
        let xy: Vec<(f64, f64)> = tops.iter().map(|&i| r.cell_center(i / r.ncols, i % r.ncols)).collect();
        assert_eq!(xy, vec![(10.25, 10.25), (22.25, 10.25)]);
        // A window wider than their spacing keeps only the taller.
        let tops = local_maxima_raster(&r, &Window::Fixed(30.0), 2.0, Shape::Square).unwrap();
        assert_eq!(tops.len(), 1);
        // hmin above the smaller cone.
        assert_eq!(local_maxima_raster(&r, &Window::Fixed(3.0), 9.0, Shape::Circular).unwrap().len(), 1);
        // A height-dependent window: 1 m for the short cone, 30 m for the tall one.
        let w = Window::Linear { intercept: -85.0, slope: 11.5, min: 1.0, max: 30.0 };
        assert_eq!(local_maxima_raster(&r, &w, 2.0, Shape::Circular).unwrap().len(), 2);
        let w = Window::Linear { intercept: 30.0, slope: 0.0, min: 1.0, max: 30.0 };
        assert_eq!(local_maxima_raster(&r, &w, 2.0, Shape::Circular).unwrap().len(), 1);
        assert!(local_maxima_raster(&r, &Window::Fixed(0.0), 2.0, Shape::Circular).is_err());
    }

    #[test]
    fn equal_maxima_keep_the_first() {
        let r = raster(&[&[0.0, 0.0, 0.0, 0.0], &[0.0, 5.0, 5.0, 0.0], &[0.0, 0.0, 0.0, 0.0]]);
        let tops = local_maxima_raster(&r, &Window::Fixed(3.0), 2.0, Shape::Circular).unwrap();
        assert_eq!(tops, vec![5]);
        // Points: same rule, by x.
        let xy = [[1.0, 1.0], [0.0, 1.0], [5.0, 5.0]];
        let h = [5.0, 5.0, 1.0];
        assert_eq!(local_maxima_points(&xy, &h, &Window::Fixed(3.0), 2.0, Shape::Circular).unwrap(), vec![1]);
        // Farther apart than the window: both.
        assert_eq!(local_maxima_points(&xy, &h, &Window::Fixed(1.5), 2.0, Shape::Circular).unwrap(), vec![1, 0]);
    }

    #[test]
    fn points_and_brute_force_agree() {
        let mut rng = crate::nprandom::Generator::new(4);
        let n = 3000;
        let v = rng.uniform_n(0.0, 30.0, 3 * n);
        let xy: Vec<[f64; 2]> = (0..n).map(|i| [v[3 * i], v[3 * i + 1]]).collect();
        let h: Vec<f64> = (0..n).map(|i| v[3 * i + 2]).collect();
        for (w, shape) in [(Window::Fixed(2.5), Shape::Circular), (Window::Linear { intercept: 0.5, slope: 0.2, min: 1.0, max: 6.0 }, Shape::Square)] {
            let got = local_maxima_points(&xy, &h, &w, 2.0, shape).unwrap();
            let mut want: Vec<usize> = (0..n)
                .filter(|&i| h[i] >= 2.0 && (0..n).all(|j| !(h[j] > h[i] && shape.contains(xy[j][0] - xy[i][0], xy[j][1] - xy[i][1], w.at(i, h[i]) / 2.0))))
                .collect();
            want.sort_by(|&a, &b| xy[a][0].total_cmp(&xy[b][0]));
            assert_eq!(got, want);
        }
    }

    #[test]
    fn watershed_splits_two_cones_at_the_valley() {
        let r = cones();
        let tops = local_maxima_raster(&r, &Window::Fixed(3.0), 2.0, Shape::Circular).unwrap();
        let t: Vec<[f64; 3]> = tops.iter().map(|&i| { let (x, y) = r.cell_center(i / r.ncols, i % r.ncols); [x, y, r.data[i]] }).collect();
        let seeds = seed_raster(&r, &t);
        let l = watershed(&r, &seeds, 2.0).unwrap();
        for (i, &v) in l.iter().enumerate() {
            let (x, y) = r.cell_center(i / r.ncols, i % r.ncols);
            let a = 10.0 - 1.2 * ((x - 10.25).hypot(y - 10.25));
            let b = 8.0 - 1.2 * ((x - 22.25).hypot(y - 10.25));
            let want = if a.max(b) <= 2.0 { 0 } else if a >= b { 1 } else { 2 };
            if (a - b).abs() > 0.8 || want == 0 {
                assert_eq!(v, want, "cell at {x} {y}");
            }
        }
    }

    #[test]
    fn dalponte_grows_within_its_thresholds() {
        let r = cones();
        let seeds = seed_raster(&r, &[[10.25, 10.25, 10.0], [22.25, 10.25, 8.0]]);
        let d = Dalponte { max_cr: 100.0, ..Default::default() };
        let l = dalponte2016(&r, &seeds, &d).unwrap();
        for (i, &v) in l.iter().enumerate() {
            let (row, col) = (i / r.ncols, i % r.ncols);
            let (x, y) = r.cell_center(row, col);
            let z = r.data[i];
            if v == 1 {
                assert!(z > 10.0 * 0.45 && z > 2.0, "{x} {y} {z}");
            }
            if v == 2 {
                assert!(z > 8.0 * 0.45, "{x} {y} {z}");
            }
        }
        // Cells well above half the seed height, near the top, are in the crown.
        let c = r.cell_index(11.0, 11.0);
        assert_eq!(l[c.0 as usize * r.ncols + c.1 as usize], 1);
        // max_cr limits the extent to fewer than max_cr cells from the seed.
        let small = dalponte2016(&r, &seeds, &Dalponte { max_cr: 3.0, ..Default::default() }).unwrap();
        let (sr, sc) = r.cell_index(10.25, 10.25);
        for (i, &v) in small.iter().enumerate() {
            if v == 1 {
                assert!(((i / r.ncols) as i64 - sr).abs() < 3 && ((i % r.ncols) as i64 - sc).abs() < 3);
            }
        }
        assert!(dalponte2016(&r, &seeds, &Dalponte { th_seed: 1.5, ..Default::default() }).is_err());
    }

    #[test]
    fn li2012_separates_two_clusters() {
        // Two cone-shaped point clusters 8 m apart.
        let mut rng = crate::nprandom::Generator::new(1);
        let v = rng.uniform_n(0.0, 1.0, 4000);
        let mut xy = Vec::new();
        let mut h = Vec::new();
        for k in 0..2000 {
            let (cx, top) = if k % 2 == 0 { (0.0, 20.0) } else { (8.0, 16.0) };
            let r = 3.0 * v[2 * k].sqrt();
            let a = 2.0 * std::f64::consts::PI * v[2 * k + 1];
            xy.push([cx + r * a.cos(), r * a.sin()]);
            h.push(top - 2.0 * r);
        }
        let l = li2012(&xy, &h, &Li2012::default()).unwrap();
        for k in 0..2000 {
            assert_eq!(l[k], if k % 2 == 0 { 1 } else { 2 }, "point {k}");
        }
        assert!(li2012(&xy, &h, &Li2012 { dt1: 0.0, ..Default::default() }).is_err());
    }

    #[test]
    fn near_set_is_exact() {
        let mut rng = crate::nprandom::Generator::new(2);
        let v = rng.uniform_n(-5.0, 5.0, 400);
        let mut s = NearSet::new(0.5, 24);
        s.reset(-6.0, -6.0);
        assert_eq!(s.nearest_sq(0.0, 0.0), f64::INFINITY);
        let pts: Vec<[f64; 2]> = v.chunks(2).take(100).map(|c| [c[0], c[1]]).collect();
        for p in &pts {
            s.push(*p);
        }
        for q in v.chunks(2).skip(100) {
            let want = pts.iter().map(|p| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).fold(f64::INFINITY, f64::min);
            assert_eq!(s.nearest_sq(q[0], q[1]), want);
        }
    }

    #[test]
    fn chi_shape_follows_a_notch() {
        // A 10 x 10 square of points with a 4 m wide notch cut from the top.
        let mut pts = Vec::new();
        for i in 0..=20 {
            for j in 0..=20 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                if !(x > 3.0 && x < 7.0 && y > 4.0) {
                    pts.push([x, y]);
                }
            }
        }
        let convex = polygon_area(&hull(&pts, Hull::Convex));
        assert!((convex - 100.0).abs() < 1e-9);
        let concave = polygon_area(&hull(&pts, Hull::Concave(1.0)));
        // The notch (3.5 x 5.5 m between the points bordering it, less its corners) is gone.
        assert!(concave < 85.0 && concave > 70.0, "{concave}");
        // Too few points: the convex hull.
        assert_eq!(hull(&pts[..2], Hull::Concave(1.0)).len(), 2);
    }

    #[test]
    fn segment_cloud_finds_two_trees() {
        let mut rng = crate::nprandom::Generator::new(3);
        let v = rng.uniform_n(0.0, 1.0, 20000);
        let mut xyz = Vec::new();
        for k in 0..10000 {
            let (cx, top) = if k % 2 == 0 { (5.0, 20.0) } else { (14.0, 15.0) };
            let r = 3.5 * v[2 * k].sqrt();
            let a = 2.0 * std::f64::consts::PI * v[2 * k + 1];
            xyz.push([cx + r * a.cos(), 5.0 + r * a.sin(), top - 2.0 * r]);
        }
        let h: Vec<f64> = xyz.iter().map(|p| p[2]).collect();
        for method in [Method::Dalponte(Dalponte::default()), Method::Watershed { th_tree: 2.0 }, Method::Li2012(Li2012::default())] {
            let p = TreeParams { method, ..Default::default() };
            let (trees, labels) = segment_cloud(&xyz, &h, &p).unwrap();
            assert_eq!(trees.len(), 2, "{method:?}");
            assert!((trees[0].x - 5.0).abs() < 0.5 && (trees[1].x - 14.0).abs() < 0.5);
            let area = std::f64::consts::PI * 3.5 * 3.5;
            for t in &trees {
                assert!((t.crown_area - area).abs() < 0.15 * area, "{method:?} {}", t.crown_area);
            }
            let right = (0..xyz.len()).filter(|&k| labels[k] == if k % 2 == 0 { 1 } else { 2 }).count();
            assert!(right as f64 > 0.97 * xyz.len() as f64, "{method:?} {right}");
        }
        let (tops, labels) = segment_cloud(&xyz, &h, &TreeParams { method: Method::Tops, tops_from: TopsFrom::Points, ..Default::default() }).unwrap();
        assert_eq!(tops.len(), 2);
        assert!(tops[0].crown_area.is_nan() && labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn a_catalogue_gives_the_trees_of_the_merged_cloud() {
        use crate::als::{read_region, Layout};
        use crate::als_ops::write_tiles;
        use crate::io::las::LasWriteOptions;
        use crate::synthetic_als::{fly, stand, FlightParams};
        let trees = stand(14, 60.0, 7.0, (10.0, 22.0), 5).unwrap();
        let scene = crate::synthetic::forest(&trees, 60.0, 100, 0.0, 1);
        let f = fly(&scene, &FlightParams { pulse_rate: 6000.0, line_spacing: 30.0, bounds: Some([0.0, 0.0, 60.0, 60.0]), seed: 2, ..Default::default() }).unwrap();
        let dir = std::env::temp_dir().join(format!("sylva-als-trees-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        #[allow(clippy::needless_update)]
        let wo = LasWriteOptions { point_format: 6, scale: 0.001, ..Default::default() };
        let tiles = write_tiles(&f.points, &dir.join("in"), 30.0, Some((0.0, 0.0)), "laz", &wo, None).unwrap();
        let cat = Catalog::open(&tiles.iter().map(|t| t.0.clone()).collect::<Vec<_>>());
        let merged = read_region(&cat, [-1.0, -1.0, 61.0, 61.0]).unwrap();
        let heights = Heights::Dtm({
            let mut r = Raster::filled(62, 62, -1.0, -1.0, 1.0, 0.0);
            for row in 0..62 {
                for col in 0..62 {
                    let (x, y) = r.cell_center(row, col);
                    r.set(row, col, crate::synthetic::terrain_height(x, y, 0.05));
                }
            }
            r
        });
        let Heights::Dtm(dtm) = &heights else { unreachable!() };
        let h = ground::heights_above(&merged.xyz, dtm);
        for method in [Method::Dalponte(Dalponte::default()), Method::Watershed { th_tree: 2.0 }, Method::Li2012(Li2012::default())] {
            let p = TreeParams { method, window: Window::Linear { intercept: 2.0, slope: 0.15, min: 3.0, max: 6.0 }, ..Default::default() };
            let (whole, _) = segment_cloud(&merged.xyz, &h, &p).unwrap();
            assert!(whole.len() >= 10, "{method:?}: {}", whole.len());
            for (layout, workers) in [(Layout::Tiles, 1), (Layout::Grid { size: 20.0, origin: None }, 3)] {
                let got = catalog_trees(&cat, &p, &heights, Some(&dir.join("out")), "tree", Some("las"), &RunOptions { layout, buffer: 25.0, workers }).unwrap();
                assert_eq!(format!("{:?}", got.trees), format!("{whole:?}"), "{method:?} {layout:?}");
                assert_eq!(got.unmatched_points, 0);
                let back: usize = got.written.iter().map(|p| crate::io::read(p).unwrap().len()).sum();
                assert_eq!(back, merged.len());
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
