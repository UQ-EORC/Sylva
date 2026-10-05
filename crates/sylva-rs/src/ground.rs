// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Ground classification, DTM, height normalisation and CHM.

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::raster::{Raster, Reducer};
use crate::{Point, PointCloud};

/// ASPRS ground class.
pub const GROUND_CLASS: u8 = 2;
pub const UNCLASSIFIED: u8 = 1;

/// Cloth Simulation Filter parameters (Zhang et al. 2016).
#[derive(Debug, Clone)]
pub struct CsfParams {
    /// Cloth grid spacing (m). ~0.5 m suits most TLS plots.
    pub cloth_resolution: f64,
    /// 1 (steep terrain) .. 3 (flat): strength of the internal spring constraint.
    pub rigidness: usize,
    /// Height above the settled cloth below which a point is ground (m).
    pub class_threshold: f64,
    pub iterations: usize,
    pub time_step: f64,
}

impl Default for CsfParams {
    fn default() -> Self {
        CsfParams { cloth_resolution: 0.5, rigidness: 2, class_threshold: 0.3, iterations: 500, time_step: 0.65 }
    }
}

/// Ground mask by cloth simulation: `true` = ground.
///
/// The method, in a sentence: turn the cloud upside down, drop a stiff cloth
/// onto it, and call ground whatever the settled cloth lies close to. The
/// cloth cannot pass through the surface, so it drapes over the terrain and
/// bridges the inverted canopy instead of sagging into it.
///
/// The simulation is the usual mass-spring one. Each cell of the cloth falls
/// under gravity, keeps a little of its previous motion, and stops when it
/// meets the surface; then the `rigidness` inner passes pull each cell back
/// towards its neighbours' heights, which is what stops the cloth following
/// every hollow. The loop ends when nothing moves appreciably, or after
/// `iterations`.
pub fn csf_ground_mask(points: &[Point], p: &CsfParams) -> Vec<bool> {
    if points.is_empty() {
        return Vec::new();
    }
    let res = p.cloth_resolution;
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for q in points {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let xmin = lo[0] - res;
    let ymin = lo[1] - res;
    let ncols = (((hi[0] + res - xmin) / res).ceil() as usize) + 1;
    let nrows = (((hi[1] + res - ymin) / res).ceil() as usize) + 1;
    let idx = |r: usize, c: usize| r * ncols + c;

    // Inverted surface: highest -z (lowest z) per cell.
    let mut surface = vec![f64::NAN; nrows * ncols];
    for q in points {
        let c = ((q[0] - xmin) / res) as usize;
        let r = ((q[1] - ymin) / res) as usize;
        let v = -q[2];
        let s = &mut surface[idx(r, c)];
        if s.is_nan() || v > *s {
            *s = v;
        }
    }
    let mut surf_r = Raster { data: surface, nrows, ncols, xmin, ymin, resolution: res };
    surf_r.fill_nearest();
    let surface = surf_r.data;

    // The cloth starts as a flat sheet below everything in the inverted
    // cloud, so it has to fall onto the surface rather than start inside it.
    // `prev` is where each cell was on the previous step, which is how the
    // motion carries over; a cell that has met the surface stops being
    // movable and is pinned there for the rest of the run.
    let inv_min = points.iter().map(|q| -q[2]).fold(f64::INFINITY, f64::min);
    let mut height = vec![inv_min - 1.0; nrows * ncols];
    let mut prev = height.clone();
    let mut movable = vec![true; nrows * ncols];
    let gravity = 0.2 * p.time_step * p.time_step;
    let mut delta = vec![0.0; nrows * ncols];

    for _ in 0..p.iterations {
        let mut max_move: f64 = 0.0;
        for i in 0..height.len() {
            let nh = if movable[i] { height[i] + (height[i] - prev[i]) * 0.99 + gravity } else { height[i] };
            prev[i] = height[i];
            height[i] = nh;
            if height[i] >= surface[i] {
                height[i] = surface[i];
                movable[i] = false;
            }
        }
        // Stiffness: each pass moves a cell halfway towards each neighbour
        // it is out of line with, and more passes make a stiffer cloth. The
        // moves are accumulated in `delta` and applied together, so that a
        // cell is pulled by its neighbours' current heights rather than by
        // whatever they have already been changed to this pass.
        for _ in 0..p.rigidness {
            delta.iter_mut().for_each(|d| *d = 0.0);
            for r in 0..nrows {
                for c in 0..ncols {
                    let i = idx(r, c);
                    if !movable[i] {
                        continue;
                    }
                    // The four neighbours. Row and column are unsigned, so
                    // one step back from row 0 wraps around to a huge number
                    // instead of going negative, and the `< nrows` test
                    // rejects it - which is why the edges need no special
                    // case. A neighbour already resting on the surface pulls
                    // at full strength, a still-falling one at half, since
                    // that one is being pulled the other way at the same
                    // time.
                    let mut acc = 0.0;
                    let nb = [(r.wrapping_sub(1), c), (r + 1, c), (r, c.wrapping_sub(1)), (r, c + 1)];
                    for (nr, nc) in nb {
                        if nr < nrows && nc < ncols {
                            let j = idx(nr, nc);
                            let factor = if movable[j] { 0.5 } else { 1.0 };
                            acc += (height[j] - height[i]) * factor;
                        }
                    }
                    delta[i] = acc / 4.0;
                }
            }
            for i in 0..height.len() {
                height[i] = (height[i] + delta[i]).min(surface[i]);
            }
        }
        for i in 0..height.len() {
            max_move = max_move.max((height[i] - prev[i]).abs());
        }
        if max_move < 1e-4 {
            break;
        }
    }

    let cloth = Raster { data: height.iter().map(|h| -h).collect(), nrows, ncols, xmin, ymin, resolution: res };
    points
        .iter()
        .map(|q| {
            // Bilinear at the cloth node grid (nodes at cell corners).
            let fx = ((q[0] - xmin) / res).clamp(0.0, (ncols - 1) as f64);
            let fy = ((q[1] - ymin) / res).clamp(0.0, (nrows - 1) as f64);
            let c0 = (fx.floor() as usize).min(ncols - 2);
            let r0 = (fy.floor() as usize).min(nrows - 2);
            let tx = (fx - c0 as f64).clamp(0.0, 1.0);
            let ty = (fy - r0 as f64).clamp(0.0, 1.0);
            let z = cloth.get(r0, c0) * (1.0 - tx) * (1.0 - ty)
                + cloth.get(r0, c0 + 1) * tx * (1.0 - ty)
                + cloth.get(r0 + 1, c0) * (1.0 - tx) * ty
                + cloth.get(r0 + 1, c0 + 1) * tx * ty;
            (q[2] - z).abs() <= p.class_threshold
        })
        .collect()
}

/// Progressive Morphological Filter parameters (Zhang et al. 2003).
#[derive(Debug, Clone)]
pub struct PmfParams {
    pub cell_size: f64,
    pub max_window: f64,
    pub slope: f64,
    pub initial_distance: f64,
    pub max_distance: f64,
}

impl Default for PmfParams {
    fn default() -> Self {
        PmfParams { cell_size: 0.5, max_window: 10.0, slope: 0.3, initial_distance: 0.15, max_distance: 2.0 }
    }
}

/// Ground mask by progressive morphological filtering.
pub fn pmf_ground_mask(points: &[Point], p: &PmfParams) -> Result<Vec<bool>> {
    let mut lowest = Raster::from_points(points.iter().map(|q| (q[0], q[1])), points.iter().map(|q| q[2]), p.cell_size, Reducer::Min, None, f64::NAN)?;
    lowest.fill_nearest();
    let mut surface = lowest.clone();
    let mut window = 1usize;
    while window as f64 * p.cell_size <= p.max_window {
        let opened = surface.grey_opening(window);
        let threshold = (p.slope * window as f64 * p.cell_size + p.initial_distance).min(p.max_distance);
        for i in 0..surface.data.len() {
            if surface.data[i] - opened.data[i] > threshold {
                surface.data[i] = opened.data[i];
            }
        }
        window *= 2;
    }
    let thr = p.initial_distance + p.slope * p.cell_size;
    Ok(points
        .iter()
        .map(|q| {
            let (r, c) = surface.cell_index(q[0], q[1]);
            surface.in_bounds(r, c) && q[2] - surface.get(r as usize, c as usize) <= thr
        })
        .collect())
}

/// Write a `classification` attribute (2 = ground, 1 = other) from a mask.
pub fn classification_from_mask(mask: &[bool]) -> Attr {
    Attr::U8(mask.iter().map(|&g| if g { GROUND_CLASS } else { UNCLASSIFIED }).collect())
}

/// Boolean ground mask from the cloud's `classification` attribute.
pub fn ground_mask(cloud: &PointCloud) -> Result<Vec<bool>> {
    let c = cloud.attr("classification").ok_or_else(|| Error::invalid("cloud has no 'classification' attribute; classify ground first"))?;
    Ok((0..cloud.len()).map(|i| c.get_f64(i) as u8 == GROUND_CLASS).collect())
}

/// DTM from ground points: lowest ground point per cell, gaps filled from the
/// nearest cell and lightly smoothed, then the terrain is re-sampled.
///
/// `bounds` is `(xmin, ymin, xmax, ymax)`.
pub fn make_dtm(ground: &[Point], resolution: f64, bounds: Option<(f64, f64, f64, f64)>) -> Result<Raster> {
    if ground.len() < 3 {
        return Err(Error::invalid("need at least 3 ground points"));
    }
    let mut dtm = Raster::from_points(ground.iter().map(|q| (q[0], q[1])), ground.iter().map(|q| q[2]), resolution, Reducer::Min, bounds, f64::NAN)?;
    let empty: Vec<bool> = dtm.data.iter().map(|v| v.is_nan()).collect();
    dtm.fill_nearest();
    // Smooth only the filled cells so measured cells keep their value.
    let smoothed = dtm.smooth(2);
    for i in 0..dtm.data.len() {
        if empty[i] {
            dtm.data[i] = smoothed.data[i];
        }
    }
    Ok(dtm)
}

/// Height of each point above the DTM.
pub fn heights_above(points: &[Point], dtm: &Raster) -> Vec<f64> {
    points.iter().map(|q| q[2] - dtm.sample(q[0], q[1])).collect()
}

/// Canopy height model: maximum height per cell, 0 where nothing is above `min_height`.
pub fn make_chm(xy: &[Point], heights: &[f64], resolution: f64, bounds: Option<(f64, f64, f64, f64)>, min_height: f64) -> Result<Raster> {
    let keep: Vec<usize> = (0..xy.len()).filter(|&i| heights[i] >= min_height).collect();
    Raster::from_points(keep.iter().map(|&i| (xy[i][0], xy[i][1])), keep.iter().map(|&i| heights[i]), resolution, Reducer::Max, bounds, 0.0)
}
