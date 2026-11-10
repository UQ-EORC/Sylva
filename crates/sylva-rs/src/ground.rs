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
/// ASPRS class for a point not identified as ground (LAS "unclassified").
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
    /// Maximum number of simulation steps; the run also stops once no cell moves more than 0.1 mm.
    pub iterations: usize,
    /// Simulation time step (dimensionless); a free cell falls 0.2 * time_step^2 m a step. The default rarely needs changing.
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
///
/// Each cloth node stands for one `cloth_resolution` cell and sits at its
/// centre, resting on the lowest point of the cell; a point's height above
/// the cloth is taken from the four nodes around it. On a slope the lowest
/// point of a cell is up to half a cell downhill of its centre, so ground
/// reads up to `slope * cloth_resolution / 2` above the cloth, whichever
/// way the slope faces: keep that below `class_threshold` on steep terrain.
///
/// A point with a NaN or infinite coordinate takes no part and is not
/// ground.
///
/// # Errors
/// `cloth_resolution`, `class_threshold` or `time_step` that is not a
/// finite number, with the first and last positive and the threshold not
/// negative.
pub fn csf_ground_mask(points: &[Point], p: &CsfParams) -> Result<Vec<bool>> {
    let res = p.cloth_resolution;
    if !(res.is_finite() && res > 0.0) {
        return Err(Error::invalid(format!("cloth_resolution must be a positive number of metres, got {res}")));
    }
    if !(p.class_threshold.is_finite() && p.class_threshold >= 0.0) {
        return Err(Error::invalid(format!("class_threshold must be a non-negative number of metres, got {}", p.class_threshold)));
    }
    if !(p.time_step.is_finite() && p.time_step > 0.0) {
        return Err(Error::invalid(format!("time_step must be a positive number, got {}", p.time_step)));
    }
    let finite = |q: &Point| q[0].is_finite() && q[1].is_finite() && q[2].is_finite();
    if !points.iter().any(finite) {
        return Ok(vec![false; points.len()]);
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for q in points.iter().filter(|q| finite(q)) {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let xmin = lo[0] - res;
    let ymin = lo[1] - res;
    let ncols = (((hi[0] + res - xmin) / res).ceil() as usize) + 1;
    let nrows = (((hi[1] + res - ymin) / res).ceil() as usize) + 1;
    crate::util::limits::check_cells(nrows as u128 * ncols as u128, 49, &format!("a {nrows} x {ncols} cloth at {res} m"), "a coarser cloth_resolution")?;
    let idx = |r: usize, c: usize| r * ncols + c;

    // Inverted surface: highest -z (lowest z) per cell.
    let mut surface = vec![f64::NAN; nrows * ncols];
    for q in points.iter().filter(|q| finite(q)) {
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
    let inv_min = points.iter().filter(|q| finite(q)).map(|q| -q[2]).fold(f64::INFINITY, f64::min);
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
    Ok(points
        .iter()
        .map(|q| {
            if !finite(q) {
                return false;
            }
            // Bilinear between the cloth nodes, each at the centre of the
            // cell whose lowest point it rests on (hence the half cell).
            let fx = ((q[0] - xmin) / res - 0.5).clamp(0.0, (ncols - 1) as f64);
            let fy = ((q[1] - ymin) / res - 0.5).clamp(0.0, (nrows - 1) as f64);
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
        .collect())
}

/// Progressive Morphological Filter parameters (Zhang et al. 2003).
#[derive(Debug, Clone)]
pub struct PmfParams {
    /// Cell size of the minimum-height grid the filter opens (m).
    pub cell_size: f64,
    /// Largest opening window (m), its full width: the filter opens with square windows of
    /// `(2 w + 1) * cell_size` for `w` = 1, 2, 4, ... cells while that width is at most
    /// `max_window`. It should exceed the footprint of the largest non-ground object (a
    /// crown, a building); below `3 * cell_size` nothing is opened.
    pub max_window: f64,
    /// Terrain slope (rise over run, dimensionless) that grows the height threshold with window size.
    pub slope: f64,
    /// Height threshold for the smallest window (m).
    pub initial_distance: f64,
    /// Upper limit on the height threshold (m).
    pub max_distance: f64,
}

impl Default for PmfParams {
    fn default() -> Self {
        PmfParams { cell_size: 0.5, max_window: 10.0, slope: 0.3, initial_distance: 0.15, max_distance: 2.0 }
    }
}

/// Ground mask by progressive morphological filtering: `true` = ground.
///
/// The lowest point of each `cell_size` cell makes a surface, which is
/// opened (eroded, then dilated) with square windows of half-width `w` = 1,
/// 2, 4, ... cells, as long as the window's full width `(2 w + 1) *
/// cell_size` is no more than `max_window`. After each opening a cell more
/// than `slope * w * cell_size + initial_distance` (at most `max_distance`)
/// above the opened surface is lowered to it; a point is ground when it is
/// within `initial_distance + slope * cell_size` of the final surface.
///
/// # Errors
/// `cell_size` or `max_window` that is not a positive, finite number;
/// `slope`, `initial_distance` or `max_distance` that is negative or not
/// finite; or no point with finite coordinates.
pub fn pmf_ground_mask(points: &[Point], p: &PmfParams) -> Result<Vec<bool>> {
    for (name, v, strict) in [("cell_size", p.cell_size, true), ("max_window", p.max_window, true), ("slope", p.slope, false), ("initial_distance", p.initial_distance, false), ("max_distance", p.max_distance, false)] {
        if !v.is_finite() || v < 0.0 || (strict && v == 0.0) {
            return Err(Error::invalid(format!("{name} must be a {} number, got {v}", if strict { "positive" } else { "non-negative" })));
        }
    }
    let mut lowest = Raster::from_points(points.iter().map(|q| (q[0], q[1])), points.iter().map(|q| q[2]), p.cell_size, Reducer::Min, None, f64::NAN)?;
    lowest.fill_nearest();
    let mut surface = lowest.clone();
    let mut window = 1usize;
    while (2 * window + 1) as f64 * p.cell_size <= p.max_window {
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

/// Canopy height model: maximum height (m) per cell of `resolution` m, 0
/// where nothing is at or above `min_height`.
///
/// `bounds` is `(xmin, ymin, xmax, ymax)`, sized as [`Raster::from_points`]
/// sizes it. Without it the grid covers every point with finite x and y,
/// whatever its height, so a cloud with little or nothing above `min_height`
/// still gives a raster of its whole extent (zeros where there is no
/// canopy) and cover computed from it refers to the whole cloud.
///
/// # Errors
/// `heights` not as long as `xy`, a `resolution` that is not positive and
/// finite, or (without `bounds`) no point with finite x and y.
pub fn make_chm(xy: &[Point], heights: &[f64], resolution: f64, bounds: Option<(f64, f64, f64, f64)>, min_height: f64) -> Result<Raster> {
    if heights.len() != xy.len() {
        return Err(Error::invalid(format!("{} heights for {} points", heights.len(), xy.len())));
    }
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(Error::invalid(format!("resolution must be a positive number of metres, got {resolution}")));
    }
    let bounds = match bounds {
        Some(b) => b,
        None => {
            // The extent of the cloud, not of the points that pass `min_height`.
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for q in xy.iter().filter(|q| q[0].is_finite() && q[1].is_finite()) {
                for k in 0..2 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            if !lo[0].is_finite() {
                return Err(Error::invalid("cannot rasterise an empty point set"));
            }
            ((lo[0] / resolution).floor() * resolution, (lo[1] / resolution).floor() * resolution, hi[0], hi[1])
        }
    };
    let bounds = Some(bounds);
    let keep: Vec<usize> = (0..xy.len()).filter(|&i| heights[i] >= min_height).collect();
    Raster::from_points(keep.iter().map(|&i| (xy[i][0], xy[i][1])), keep.iter().map(|&i| heights[i]), resolution, Reducer::Max, bounds, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproducible uniform numbers in `[0, 1)`.
    fn uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Share of the points away from the edge that `mask` calls ground.
    fn inner_share(pts: &[Point], mask: &[bool], lo: Point, side: f64) -> f64 {
        let inner: Vec<bool> = pts.iter().zip(mask).filter(|(p, _)| p[0] > lo[0] + 8.0 && p[0] < lo[0] + side - 8.0 && p[1] > lo[1] + 8.0 && p[1] < lo[1] + side - 8.0).map(|(_, &m)| m).collect();
        inner.iter().filter(|&&m| m).count() as f64 / inner.len() as f64
    }

    #[test]
    fn csf_ground_does_not_depend_on_the_aspect() {
        // A bare 20 degree plane in UTM-scale coordinates, facing each of
        // the four ways: all of it is ground whichever way it faces.
        let o = [512_000.0, 6_234_000.0, 300.0];
        for (axis, g) in [(0, 0.36), (0, -0.36), (1, 0.36), (1, -0.36)] {
            let mut u = uniform(11);
            let pts: Vec<Point> = (0..120_000)
                .map(|_| {
                    let (x, y) = (60.0 * u(), 60.0 * u());
                    [o[0] + x, o[1] + y, o[2] + g * [x, y][axis] + 0.04 * (u() - 0.5)]
                })
                .collect();
            let mask = csf_ground_mask(&pts, &CsfParams { cloth_resolution: 1.0, ..Default::default() }).unwrap();
            let share = inner_share(&pts, &mask, o, 60.0);
            assert!(share > 0.99, "slope {g} along axis {axis}: {share}");
        }
    }

    #[test]
    fn csf_refuses_bad_settings_and_skips_points_without_a_position() {
        let o = [512_000.0, 6_234_000.0, 300.0];
        let mut u = uniform(5);
        let mut pts: Vec<Point> = (0..20_000).map(|_| [o[0] + 30.0 * u(), o[1] + 30.0 * u(), o[2] + 0.02 * u()]).collect();
        let clean = csf_ground_mask(&pts, &CsfParams::default()).unwrap();
        pts.push([f64::NAN, o[1], o[2]]);
        pts.push([o[0], f64::INFINITY, o[2]]);
        pts.push([o[0], o[1], f64::NAN]);
        let mask = csf_ground_mask(&pts, &CsfParams::default()).unwrap();
        assert_eq!(mask.len(), pts.len());
        assert_eq!(mask[..clean.len()], clean[..]);
        assert_eq!(mask[clean.len()..], [false; 3]);
        for res in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            assert!(csf_ground_mask(&pts, &CsfParams { cloth_resolution: res, ..Default::default() }).is_err(), "{res}");
        }
        assert!(csf_ground_mask(&[], &CsfParams::default()).unwrap().is_empty());
        assert_eq!(csf_ground_mask(&[[f64::NAN; 3]], &CsfParams::default()).unwrap(), [false]);
    }

    #[test]
    fn pmf_window_is_a_full_width_in_metres() {
        // Flat ground with a 12 m wide flat roof 5 m up and nothing under it.
        let o = [512_000.0, 6_234_000.0, 300.0];
        let mut u = uniform(3);
        let mut pts = Vec::new();
        let mut roof = Vec::new();
        for _ in 0..200_000 {
            let (x, y) = (60.0 * u(), 60.0 * u());
            let on = (24.0..36.0).contains(&x) && (24.0..36.0).contains(&y);
            pts.push([o[0] + x, o[1] + y, o[2] + if on { 5.0 } else { 0.0 }]);
            roof.push(on);
        }
        let roof_ground = |max_window: f64| {
            let m = pmf_ground_mask(&pts, &PmfParams { max_window, ..Default::default() }).unwrap();
            let n = roof.iter().filter(|&&r| r).count() as f64;
            m.iter().zip(&roof).filter(|(&g, &r)| g && r).count() as f64 / n
        };
        // A 10 m window cannot span the roof, so it stays in the surface; 20 m removes it.
        assert!(roof_ground(10.0) > 0.9, "{}", roof_ground(10.0));
        assert_eq!(roof_ground(20.0), 0.0);
        let m = pmf_ground_mask(&pts, &PmfParams { max_window: 20.0, ..Default::default() }).unwrap();
        assert!(m.iter().zip(&roof).filter(|(_, &r)| !r).all(|(&g, _)| g));
    }

    #[test]
    fn pmf_refuses_sizes_that_would_never_end() {
        let pts = [[512_000.0, 6_234_000.0, 300.0], [512_001.0, 6_234_001.0, 300.0]];
        let d = PmfParams::default();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(pmf_ground_mask(&pts, &PmfParams { cell_size: bad, ..d.clone() }).is_err(), "cell_size {bad}");
            assert!(pmf_ground_mask(&pts, &PmfParams { max_window: bad, ..d.clone() }).is_err(), "max_window {bad}");
        }
        for bad in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(pmf_ground_mask(&pts, &PmfParams { slope: bad, ..d.clone() }).is_err(), "slope {bad}");
        }
        assert_eq!(pmf_ground_mask(&pts, &PmfParams { slope: 0.0, ..d }).unwrap(), [true, true]);
    }

    #[test]
    fn chm_covers_the_cloud_not_the_canopy() {
        // 100 x 100 m of ground with canopy in one 20 m patch.
        let o = [512_000.0, 6_234_000.0];
        let mut u = uniform(9);
        let mut xyz = Vec::new();
        let mut h = Vec::new();
        for _ in 0..50_000 {
            let (x, y) = (100.0 * u(), 100.0 * u());
            xyz.push([o[0] + x, o[1] + y, 0.0]);
            h.push(if (40.0..60.0).contains(&x) && (40.0..60.0).contains(&y) { 15.0 } else { 0.1 });
        }
        let chm = make_chm(&xyz, &h, 1.0, None, 2.0).unwrap();
        assert_eq!((chm.nrows, chm.ncols, chm.xmin, chm.ymin), (100, 100, o[0], o[1]));
        let cover = chm.data.iter().filter(|&&v| v > 2.0).count() as f64 / chm.data.len() as f64;
        assert!((cover - 0.04).abs() < 0.005, "{cover}");
        // Nothing above the threshold is a CHM of zeros, not an error.
        let bare = make_chm(&xyz, &h, 1.0, None, 50.0).unwrap();
        assert_eq!((bare.nrows, bare.ncols), (100, 100));
        assert!(bare.data.iter().all(|&v| v == 0.0));
        assert!(make_chm(&xyz, &h, 0.0, None, 2.0).is_err());
        assert!(make_chm(&xyz, &h[..10], 1.0, None, 2.0).is_err());
        assert!(make_chm(&[], &[], 1.0, None, 2.0).is_err());
    }
}
