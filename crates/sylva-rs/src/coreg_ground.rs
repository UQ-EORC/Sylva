// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Raster ground model for scan co-registration, a faithful port of
//! `tlsalign.ground` (`fit_ground`, `GroundModel.height_at` / `support`).
//!
//! The estimator takes a low percentile of Z per cell, rejects deep pits
//! (multipath echoes metres below the terrain), fills unobserved cells from the
//! Euclidean-nearest observed cell, removes spikes with a grey opening,
//! smooths with a box filter and finally caps the slope between neighbours.
//!
//! Every step reproduces the `numpy` / `scipy.ndimage` operation tlsalign
//! uses, including scipy's `"nearest"` boundary mode (edge replication), its
//! window origin convention (`size // 2` cells on the left), the running-sum
//! arithmetic of `uniform_filter1d`, the tie-breaking of
//! `distance_transform_edt(return_indices=True)`, and `np.allclose`'s default
//! relative tolerance in the slope-limit loop. Results match tlsalign to the
//! last bit when thinning is off; the only intended difference is the random
//! thinning above `max_points`, which uses a seeded SplitMix64 selection
//! sample instead of numpy's `default_rng(seed).choice`.
//!
//! Grids are row-major `(ny, nx)`: index `iy * nx + ix`, `y` along rows.

use rayon::prelude::*;

use crate::error::{Error, Result};

/// Parameters of [`fit_ground`]; defaults are tlsalign's.
#[derive(Debug, Clone)]
pub struct GroundParams {
    /// DTM resolution (m).
    pub cell_size: f64,
    /// Per-cell Z percentile taken as the ground candidate (0 = minimum).
    pub percentile: f64,
    /// Maximum rise/run between adjacent cells; non-finite or <= 0 disables.
    pub max_slope: f64,
    /// Box-filter window (cells); <= 1 disables.
    pub smooth_cells: usize,
    /// Grey-opening window (cells); <= 1 disables.
    pub opening_cells: usize,
    /// A cell is observed when it holds at least this many points.
    pub min_points_per_cell: usize,
    /// A cell this far below its neighbourhood median is a pit; <= 0 disables.
    pub pit_depth: f64,
    /// Median window (cells) of the pit test; <= 1 disables.
    pub pit_window: usize,
    /// Fit from at most this many points, sampled at random; `None` = all.
    pub max_points: Option<usize>,
    /// Seed of the thinning sample.
    pub seed: u64,
}

impl Default for GroundParams {
    fn default() -> Self {
        GroundParams {
            cell_size: 0.5,
            percentile: 5.0,
            max_slope: 1.0,
            smooth_cells: 3,
            opening_cells: 5,
            min_points_per_cell: 1,
            pit_depth: 3.0,
            pit_window: 9,
            max_points: Some(8_000_000),
            seed: 0,
        }
    }
}

/// A raster terrain model with bilinear interpolation.
///
/// `origin` is `lo = min(xy) - cell_size` of the fitted cloud. Cell indices
/// were assigned by truncating `(p - lo) / cell_size`, so `origin` is really
/// the lower-left *corner* of cell `[0, 0]`; but, as in tlsalign, queries treat
/// it as the cell *centre*, which shifts the surface by half a cell. This is
/// kept on purpose for parity.
#[derive(Debug, Clone)]
pub struct GroundModel {
    pub nx: usize,
    pub ny: usize,
    /// `(ny, nx)` row-major terrain height, NaN-free.
    pub elevation: Vec<f64>,
    /// `(x0, y0)`.
    pub origin: [f64; 2],
    pub cell_size: f64,
    /// `(ny, nx)` row-major: cell backed by real ground points.
    pub observed: Vec<bool>,
}

impl GroundModel {
    /// Bilinear terrain height at one location (tlsalign `_height_at_block`).
    #[inline]
    pub fn height_at_point(&self, x: f64, y: f64) -> f64 {
        height_at_grid(&self.elevation, self.nx, self.ny, self.origin, self.cell_size, x, y)
    }

    /// Bilinear terrain height at many locations (parallel).
    pub fn height_at(&self, xy: &[[f64; 2]]) -> Vec<f64> {
        height_at_many(&self.elevation, self.nx, self.ny, self.origin, self.cell_size, xy)
    }

    /// Were the queried locations backed by ground returns?
    pub fn support(&self, xy: &[[f64; 2]]) -> Vec<bool> {
        support_many(&self.observed, self.nx, self.ny, self.origin, self.cell_size, xy)
    }
}

/// Bilinear interpolation on a `(ny, nx)` grid whose cell `[0, 0]` centre is
/// `origin`; locations are clamped to the grid (tlsalign `_height_at_block`).
#[inline]
pub fn height_at_grid(e: &[f64], nx: usize, ny: usize, origin: [f64; 2], cs: f64, x: f64, y: f64) -> f64 {
    let fx = ((x - origin[0]) / cs).clamp(0.0, nx as f64 - 1.0);
    let fy = ((y - origin[1]) / cs).clamp(0.0, ny as f64 - 1.0);
    if fx.is_nan() || fy.is_nan() {
        return f64::NAN;
    }
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let top = e[y0 * nx + x0] * (1.0 - tx) + e[y0 * nx + x1] * tx;
    let bottom = e[y1 * nx + x0] * (1.0 - tx) + e[y1 * nx + x1] * tx;
    top * (1.0 - ty) + bottom * ty
}

/// [`height_at_grid`] over many locations, in parallel.
pub fn height_at_many(e: &[f64], nx: usize, ny: usize, origin: [f64; 2], cs: f64, xy: &[[f64; 2]]) -> Vec<f64> {
    assert_eq!(e.len(), nx * ny, "elevation grid size mismatch");
    let mut out = vec![0.0; xy.len()];
    out.par_chunks_mut(1 << 16).zip(xy.par_chunks(1 << 16)).for_each(|(o, q)| {
        for (v, p) in o.iter_mut().zip(q) {
            *v = height_at_grid(e, nx, ny, origin, cs, p[0], p[1]);
        }
    });
    out
}

/// Nearest-cell lookup of `observed` (tlsalign `GroundModel.support`): numpy's
/// `round` (half to even), clipped to the grid.
pub fn support_many(observed: &[bool], nx: usize, ny: usize, origin: [f64; 2], cs: f64, xy: &[[f64; 2]]) -> Vec<bool> {
    assert_eq!(observed.len(), nx * ny, "observed grid size mismatch");
    let mut out = vec![false; xy.len()];
    out.par_chunks_mut(1 << 16).zip(xy.par_chunks(1 << 16)).for_each(|(o, q)| {
        for (v, p) in o.iter_mut().zip(q) {
            let ix = ((p[0] - origin[0]) / cs).round_ties_even().clamp(0.0, nx as f64 - 1.0);
            let iy = ((p[1] - origin[1]) / cs).round_ties_even().clamp(0.0, ny as f64 - 1.0);
            // NaN casts to 0, as numpy's cast would give a (bogus) index too.
            *v = observed[iy as usize * nx + ix as usize];
        }
    });
    out
}

/// Mean terrain slope (degrees) of a `(ny, nx)` row-major grid, a sanity
/// check on a fit: `np.gradient` with spacing `cs` along both axes, then
/// `degrees(arctan(hypot(gx, gy)))` averaged over the cells.
pub fn slope_deg(e: &[f64], ny: usize, nx: usize, cs: f64) -> Result<f64> {
    if e.len() != nx * ny {
        return Err(Error::invalid("elevation grid size mismatch"));
    }
    if nx < 2 || ny < 2 {
        return Err(Error::invalid("a slope needs a grid of at least 2 x 2 cells"));
    }
    let mut gx = vec![0.0; nx * ny];
    let mut gy = vec![0.0; nx * ny];
    for r in 0..ny {
        gx[r * nx..(r + 1) * nx].copy_from_slice(&crate::numeric::gradient(&e[r * nx..(r + 1) * nx], cs));
    }
    for c in 0..nx {
        let col: Vec<f64> = (0..ny).map(|r| e[r * nx + c]).collect();
        for (r, g) in crate::numeric::gradient(&col, cs).into_iter().enumerate() {
            gy[r * nx + c] = g;
        }
    }
    let deg: Vec<f64> = gx.iter().zip(&gy).map(|(x, y)| x.hypot(*y).atan().to_degrees()).collect();
    Ok(crate::coreg::numpy_sum(&deg) / deg.len() as f64)
}

/// Fit a raster DTM to a point cloud (tlsalign `fit_ground`).
pub fn fit_ground(points: &[[f64; 3]], p: &GroundParams) -> Result<GroundModel> {
    if points.is_empty() {
        return Err(Error::invalid("cannot fit a ground model to an empty cloud"));
    }
    if p.cell_size.is_nan() || p.cell_size <= 0.0 {
        return Err(Error::invalid("cell_size must be positive"));
    }
    let cs = p.cell_size;

    // The extent comes from the full cloud, so thinning cannot shrink it.
    let (mn, mx) = points
        .par_iter()
        .fold(
            || ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
            |(mut a, mut b), q| {
                for k in 0..2 {
                    a[k] = a[k].min(q[k]);
                    b[k] = b[k].max(q[k]);
                }
                (a, b)
            },
        )
        .reduce(
            || ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
            |(a, b), (c, d)| ([a[0].min(c[0]), a[1].min(c[1])], [b[0].max(d[0]), b[1].max(d[1])]),
        );
    let lo = [mn[0] - cs, mn[1] - cs];
    let hi = [mx[0] + cs, mx[1] + cs];

    let subset: Option<Vec<usize>> = match p.max_points {
        Some(m) if points.len() > m => Some(sample_sorted(points.len(), m, p.seed)),
        _ => None,
    };
    let nx = ((((hi[0] - lo[0]) / cs).ceil() as i64) + 1).max(2) as usize;
    let ny = ((((hi[1] - lo[1]) / cs).ceil() as i64) + 1).max(2) as usize;
    let ncell = nx * ny;

    // Flat cell index per (used) point: numpy astype(int64) truncates.
    let cell_of = |q: &[f64; 3]| -> usize {
        let ix = (((q[0] - lo[0]) / cs) as i64).clamp(0, nx as i64 - 1) as usize;
        let iy = (((q[1] - lo[1]) / cs) as i64).clamp(0, ny as i64 - 1) as usize;
        iy * nx + ix
    };
    let n_used = subset.as_ref().map_or(points.len(), |s| s.len());
    let point = |k: usize| -> &[f64; 3] {
        match &subset {
            Some(s) => &points[s[k]],
            None => &points[k],
        }
    };
    let mut flat = vec![0u32; n_used];
    let flat_ok = ncell <= u32::MAX as usize;
    if !flat_ok {
        return Err(Error::invalid("ground grid is too large"));
    }
    flat.par_chunks_mut(1 << 16).enumerate().for_each(|(c, chunk)| {
        let base = c << 16;
        for (j, f) in chunk.iter_mut().enumerate() {
            *f = cell_of(point(base + j)) as u32;
        }
    });

    // Counting sort of z by cell, then a per-cell order statistic.
    let mut counts = vec![0usize; ncell];
    for &f in &flat {
        counts[f as usize] += 1;
    }
    let mut observed: Vec<bool> = counts.iter().map(|&c| c >= p.min_points_per_cell).collect();
    let mut start = vec![0usize; ncell + 1];
    for i in 0..ncell {
        start[i + 1] = start[i] + counts[i];
    }
    let mut zs = vec![0.0f64; n_used];
    {
        let mut cursor = start[..ncell].to_vec();
        for (k, &f) in flat.iter().enumerate() {
            let c = &mut cursor[f as usize];
            zs[*c] = point(k)[2];
            *c += 1;
        }
    }
    drop(flat);
    let mut buckets: Vec<&mut [f64]> = Vec::with_capacity(ncell);
    {
        let mut rest: &mut [f64] = &mut zs;
        for &c in &counts {
            let (a, b) = rest.split_at_mut(c);
            buckets.push(a);
            rest = b;
        }
    }
    let pct = p.percentile;
    let mut grid: Vec<f64> = buckets
        .into_par_iter()
        .map(|b| {
            let size = b.len();
            if size == 0 {
                return f64::NAN;
            }
            let off = ((size as f64 * pct / 100.0).floor() as i64).min(size as i64 - 1).max(0) as usize;
            let (_, v, _) = b.select_nth_unstable_by(off, |a, b| a.total_cmp(b));
            *v
        })
        .collect();
    drop(zs);

    if p.pit_depth > 0.0 && p.pit_window > 1 {
        reject_pits(&mut grid, &mut observed, ny, nx, p.pit_depth, p.pit_window)?;
    }

    let mut filled = fill_nearest(&grid, &observed, ny, nx)?;
    if p.opening_cells > 1 {
        let opened = grey_opening(&filled, ny, nx, p.opening_cells);
        for (f, o) in filled.iter_mut().zip(&opened) {
            *f = np_minimum(*f, *o + 0.0);
        }
    }
    if p.smooth_cells > 1 {
        filled = uniform_filter(&filled, ny, nx, p.smooth_cells);
    }
    let elevation = enforce_max_slope(filled, ny, nx, cs, p.max_slope);
    Ok(GroundModel { nx, ny, elevation, origin: lo, cell_size: cs, observed })
}

/// `np.minimum`: propagates NaN.
#[inline]
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// Sorted random sample of `k` distinct indices from `0..n` (Knuth's
/// selection sampling, algorithm S) driven by SplitMix64.
fn sample_sorted(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut state = seed ^ 0x5EED_6A0D_D1CE_u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(k);
    let mut need = k;
    for i in 0..n {
        if need == 0 {
            break;
        }
        let left = (n - i) as u64;
        // Take i with probability need / left (exact, via 128-bit multiply).
        let r = ((next() as u128 * left as u128) >> 64) as u64;
        if r < need as u64 {
            out.push(i);
            need -= 1;
        }
    }
    out
}

/// tlsalign `_reject_pits`: two passes marking cells more than `depth` below
/// the `window` median of the nearest-filled grid as unobserved (NaN).
fn reject_pits(grid: &mut [f64], observed: &mut [bool], ny: usize, nx: usize, depth: f64, window: usize) -> Result<()> {
    for _ in 0..2 {
        let nvalid = observed.iter().zip(grid.iter()).filter(|(o, g)| **o && g.is_finite()).count();
        if nvalid < 4 {
            break;
        }
        let reference = median_filter(&fill_nearest(grid, observed, ny, nx)?, ny, nx, window);
        let mut any = false;
        for i in 0..grid.len() {
            let valid = observed[i] && grid[i].is_finite();
            if valid && grid[i] < reference[i] - depth {
                observed[i] = false;
                grid[i] = f64::NAN;
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    Ok(())
}

/// tlsalign `_fill_nearest`: every cell that is not (observed and finite)
/// takes the value of its Euclidean-nearest such cell, with scipy's
/// `distance_transform_edt(return_indices=True)` tie-breaking.
pub fn fill_nearest(grid: &[f64], observed: &[bool], ny: usize, nx: usize) -> Result<Vec<f64>> {
    let valid: Vec<bool> = observed.iter().zip(grid).map(|(o, g)| *o && g.is_finite()).collect();
    if !valid.iter().any(|&v| v) {
        return Err(Error::invalid("no ground cells were observed"));
    }
    if valid.iter().all(|&v| v) {
        return Ok(grid.to_vec());
    }
    let ft = feature_transform(&valid, ny, nx);
    Ok(ft.iter().map(|f| grid[f[0] as usize * nx + f[1] as usize]).collect())
}

/// scipy `NI_EuclideanFeatureTransform` in 2-D: for every cell, the
/// `(row, col)` of the nearest `true` ("background") cell of `features`.
/// A transcription of `_ComputeFT` / `_VoronoiFT` (ni_morphology.c), so ties
/// resolve exactly as in scipy: first a Voronoi pass down every column
/// (axis 0), then one along every row (axis 1). The algorithm is Maurer et
/// al. (2003)'s exact linear-time Euclidean distance transform.
///
/// `feature_transform` and `voronoi_line` are translated from scipy's
/// `scipy/ndimage/src/ni_morphology.c`, which carries this notice:
///
/// Copyright (C) 2003-2005 Peter J. Verveer
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// 1. Redistributions of source code must retain the above copyright
///    notice, this list of conditions and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above
///    copyright notice, this list of conditions and the following
///    disclaimer in the documentation and/or other materials provided
///    with the distribution.
///
/// 3. The name of the author may not be used to endorse or promote
///    products derived from this software without specific prior
///    written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
/// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
/// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
/// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
/// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
/// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
/// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
/// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
pub fn feature_transform(background: &[bool], ny: usize, nx: usize) -> Vec<[i32; 2]> {
    // Pass 1: columns. Stored column-major while working.
    let mut cols: Vec<[i32; 2]> = vec![[0, 0]; ny * nx];
    cols.par_chunks_mut(ny).enumerate().for_each(|(c, line)| {
        for (r, f) in line.iter_mut().enumerate() {
            *f = if background[r * nx + c] { [r as i32, c as i32] } else { [-1, 0] };
        }
        voronoi_line(line, [0, c as i64], 0);
    });
    // Pass 2: rows.
    let mut out: Vec<[i32; 2]> = vec![[0, 0]; ny * nx];
    out.par_chunks_mut(nx).enumerate().for_each(|(r, line)| {
        for (c, f) in line.iter_mut().enumerate() {
            *f = cols[c * ny + r];
        }
        voronoi_line(line, [r as i64, 0], 1);
    });
    out
}

/// scipy `_VoronoiFT` for rank 2 along axis `d`, without sampling.
#[allow(clippy::needless_range_loop)]
fn voronoi_line(line: &mut [[i32; 2]], coor: [i64; 2], d: usize) {
    let len = line.len();
    let f: Vec<[i32; 2]> = line.to_vec();
    let o = 1 - d;
    let mut g = vec![0usize; len];
    let mut l: isize = -1;
    for ii in 0..len {
        if f[ii][0] < 0 {
            continue;
        }
        let fd = f[ii][d] as f64;
        let tw = (f[ii][o] as i64 - coor[o]) as f64;
        let w_r = tw * tw;
        while l >= 1 {
            let idx1 = g[l as usize];
            let idx2 = g[l as usize - 1];
            let f1 = f[idx1][d] as f64;
            let a = f1 - f[idx2][d] as f64;
            let b = fd - f1;
            let tu = (f[idx2][o] as i64 - coor[o]) as f64;
            let tv = (f[idx1][o] as i64 - coor[o]) as f64;
            let u_r = tu * tu;
            let v_r = tv * tv;
            let c = a + b;
            if c * v_r - b * u_r - a * w_r - a * b * c <= 0.0 {
                break;
            }
            l -= 1;
        }
        l += 1;
        g[l as usize] = ii;
    }
    let maxl = l;
    if maxl < 0 {
        return;
    }
    let dist = |fe: [i32; 2], ii: usize| -> f64 {
        let mut s = 0.0;
        for jj in 0..2 {
            let t = if jj == d { (fe[jj] as i64 - ii as i64) as f64 } else { (fe[jj] as i64 - coor[jj]) as f64 };
            s += t * t;
        }
        s
    };
    let mut l = 0usize;
    for ii in 0..len {
        let mut delta1 = dist(f[g[l]], ii);
        while (l as isize) < maxl {
            let delta2 = dist(f[g[l + 1]], ii);
            if delta1 <= delta2 {
                break;
            }
            delta1 = delta2;
            l += 1;
        }
        line[ii] = f[g[l]];
    }
}

/// Offset range of a scipy window of `size` with `origin`: the window of
/// output `i` covers input `i - left .. i - left + size`.
#[inline]
fn window_left(size: usize, origin: isize) -> isize {
    size as isize / 2 + origin
}

/// scipy `minimum_filter1d` / `maximum_filter1d`, mode "nearest". The
/// grids are NaN-free here, so min/max order does not matter.
fn min_max_1d(src: &[f64], ny: usize, nx: usize, axis: usize, size: usize, origin: isize, max: bool) -> Vec<f64> {
    let left = window_left(size, origin);
    let pick = |a: f64, b: f64| if (max && b > a) || (!max && b < a) { b } else { a };
    let mut out = vec![0.0; ny * nx];
    out.par_chunks_mut(nx).enumerate().for_each(|(r, row)| {
        if axis == 0 {
            // Row r takes the element-wise extreme of rows r - left .. r - left + size.
            for j in 0..size {
                let rr = (r as isize - left + j as isize).clamp(0, ny as isize - 1) as usize;
                let s = &src[rr * nx..(rr + 1) * nx];
                if j == 0 {
                    row.copy_from_slice(s);
                } else {
                    for (o, &v) in row.iter_mut().zip(s) {
                        *o = pick(*o, v);
                    }
                }
            }
        } else {
            let s = &src[r * nx..(r + 1) * nx];
            for (c, o) in row.iter_mut().enumerate() {
                let mut acc = s[(c as isize - left).clamp(0, nx as isize - 1) as usize];
                for j in 1..size {
                    acc = pick(acc, s[(c as isize - left + j as isize).clamp(0, nx as isize - 1) as usize]);
                }
                *o = acc;
            }
        }
    });
    out
}

/// scipy `minimum_filter(size, mode="nearest", origin)` (separable).
fn minimum_filter(src: &[f64], ny: usize, nx: usize, size: usize, origin: isize) -> Vec<f64> {
    let a = min_max_1d(src, ny, nx, 0, size, origin, false);
    min_max_1d(&a, ny, nx, 1, size, origin, false)
}

/// scipy `grey_opening(size, mode="nearest")`: erosion, then dilation with
/// the reflected window (`grey_dilation` negates the origin and shifts it by
/// one for even sizes).
fn grey_opening(src: &[f64], ny: usize, nx: usize, size: usize) -> Vec<f64> {
    let eroded = minimum_filter(src, ny, nx, size, 0);
    let origin = if size.is_multiple_of(2) { -1 } else { 0 };
    let a = min_max_1d(&eroded, ny, nx, 0, size, origin, true);
    min_max_1d(&a, ny, nx, 1, size, origin, true)
}

/// scipy `uniform_filter1d`, mode "nearest", reproducing its running sum.
fn uniform_1d(src: &[f64], ny: usize, nx: usize, axis: usize, size: usize) -> Vec<f64> {
    let left = window_left(size, 0);
    let (len, lines) = if axis == 0 { (ny, nx) } else { (nx, ny) };
    let idx = |line: usize, k: usize| if axis == 0 { k * nx + line } else { line * nx + k };
    let fs = size as f64;
    let results: Vec<Vec<f64>> = (0..lines)
        .into_par_iter()
        .map(|line| {
            let ext: Vec<f64> = (0..len + size - 1)
                .map(|j| src[idx(line, (j as isize - left).clamp(0, len as isize - 1) as usize)])
                .collect();
            let mut res = Vec::with_capacity(len);
            let mut tmp = 0.0;
            for &v in &ext[..size] {
                tmp += v;
            }
            res.push(tmp / fs);
            for ll in 1..len {
                tmp += ext[ll + size - 1] - ext[ll - 1];
                res.push(tmp / fs);
            }
            res
        })
        .collect();
    let mut out = vec![0.0; ny * nx];
    for (line, res) in results.into_iter().enumerate() {
        for (k, v) in res.into_iter().enumerate() {
            out[idx(line, k)] = v;
        }
    }
    out
}

/// scipy `uniform_filter(size, mode="nearest")`: axis 0, then axis 1.
fn uniform_filter(src: &[f64], ny: usize, nx: usize, size: usize) -> Vec<f64> {
    let a = uniform_1d(src, ny, nx, 0, size);
    uniform_1d(&a, ny, nx, 1, size)
}

/// scipy `median_filter(size, mode="nearest")` in 2-D: element `n // 2` of
/// the sorted `size x size` window.
fn median_filter(src: &[f64], ny: usize, nx: usize, size: usize) -> Vec<f64> {
    let left = window_left(size, 0);
    let rank = size * size / 2;
    let mut out = vec![0.0; ny * nx];
    out.par_chunks_mut(nx).enumerate().for_each_init(
        || Vec::with_capacity(size * size),
        |buf, (r, row)| {
            for (c, o) in row.iter_mut().enumerate() {
                buf.clear();
                for a in 0..size {
                    let rr = (r as isize - left + a as isize).clamp(0, ny as isize - 1) as usize;
                    for b in 0..size {
                        let cc = (c as isize - left + b as isize).clamp(0, nx as isize - 1) as usize;
                        buf.push(src[rr * nx + cc]);
                    }
                }
                let (_, v, _) = buf.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
                *o = *v;
            }
        },
    );
    out
}

/// tlsalign `_enforce_max_slope`: up to 64 rounds of
/// `min(out, minimum_filter(out, 3) + max_slope * cell)`, stopping when
/// `np.allclose(capped, out, atol=1e-9)` (with numpy's default `rtol=1e-5`).
fn enforce_max_slope(grid: Vec<f64>, ny: usize, nx: usize, cs: f64, max_slope: f64) -> Vec<f64> {
    if !max_slope.is_finite() || max_slope <= 0.0 {
        return grid;
    }
    let step = max_slope * cs;
    let mut out = grid;
    for _ in 0..64 {
        let m = minimum_filter(&out, ny, nx, 3, 0);
        let capped: Vec<f64> = out.par_iter().zip(&m).map(|(&o, &mm)| np_minimum(o, mm + step)).collect();
        let close = capped.par_iter().zip(&out).all(|(&a, &b)| {
            if a == b {
                return true;
            }
            (a - b).abs() <= 1e-9 + 1e-5 * b.abs()
        });
        if close {
            break;
        }
        out = capped;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slope_of_a_plane() {
        let (ny, nx, cs) = (4, 5, 0.5);
        let e: Vec<f64> = (0..ny * nx).map(|k| 0.5 * cs * (k % nx) as f64).collect();
        assert!((slope_deg(&e, ny, nx, cs).unwrap() - 0.5f64.atan().to_degrees()).abs() < 1e-12);
        assert!(slope_deg(&e[..5], 1, 5, cs).is_err());
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    fn params() -> GroundParams {
        GroundParams { max_points: None, ..Default::default() }
    }

    fn cloud(f: impl Fn(f64, f64) -> f64, n: usize) -> Vec<[f64; 3]> {
        let mut s = 7;
        (0..n)
            .map(|_| {
                let x = lcg(&mut s) * 20.0;
                let y = lcg(&mut s) * 20.0;
                [x, y, f(x, y)]
            })
            .collect()
    }

    #[test]
    fn coreg_ground_flat_plane() {
        let pts = cloud(|_, _| 3.0, 40_000);
        let g = fit_ground(&pts, &params()).unwrap();
        assert!(g.elevation.iter().all(|&e| (e - 3.0).abs() < 1e-9));
        let h = g.height_at(&[[5.0, 5.0], [-100.0, 100.0]]);
        assert!((h[0] - 3.0).abs() < 1e-9 && (h[1] - 3.0).abs() < 1e-9);
        let s = g.support(&[[10.0, 10.0], [-100.0, -100.0]]);
        assert!(s[0] && !s[1]);
        assert_eq!(g.origin, [pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - 0.5, pts.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - 0.5]);
    }

    #[test]
    fn coreg_ground_slope() {
        let pts = cloud(|x, _| 0.3 * x, 80_000);
        let g = fit_ground(&pts, &params()).unwrap();
        // Interior cell centres (in tlsalign's half-cell-shifted frame) sit on
        // the plane within the cell's percentile spread.
        for iy in 4..g.ny - 4 {
            for ix in 4..g.nx - 4 {
                let x = g.origin[0] + (ix as f64 + 0.5) * g.cell_size;
                let e = g.elevation[iy * g.nx + ix];
                assert!((e - 0.3 * x).abs() < 0.1, "{ix},{iy}: {e} vs {}", 0.3 * x);
            }
        }
    }

    #[test]
    fn coreg_ground_pit_rejection() {
        let mut pts = cloud(|_, _| 0.0, 40_000);
        for k in 0..50 {
            pts.push([10.1 + 0.001 * k as f64, 10.1, -25.0]);
        }
        let g = fit_ground(&pts, &params()).unwrap();
        assert!(g.elevation.iter().all(|&e| e.abs() < 1e-9));
        let ix = ((10.1 - g.origin[0]) / 0.5) as usize;
        let iy = ((10.1 - g.origin[1]) / 0.5) as usize;
        assert!(!g.observed[iy * g.nx + ix]);
        // Without the pit test the hole drags the surface down.
        let g2 = fit_ground(&pts, &GroundParams { pit_depth: 0.0, ..params() }).unwrap();
        assert!(g2.elevation.iter().any(|&e| e < -1.0));
    }

    #[test]
    fn coreg_ground_fill_nearest() {
        // scipy's docstring example of distance_transform_edt(return_indices).
        let a = [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];
        let bg: Vec<bool> = a.iter().flatten().map(|&v| v == 0).collect();
        let ft = feature_transform(&bg, 5, 5);
        let rows = [[0, 0, 1, 1, 3], [1, 1, 1, 1, 3], [2, 2, 1, 3, 3], [3, 3, 4, 4, 3], [4, 4, 4, 4, 4]];
        let cols = [[0, 0, 1, 1, 4], [0, 1, 1, 1, 4], [0, 0, 1, 4, 4], [0, 0, 3, 3, 4], [0, 0, 3, 3, 4]];
        for r in 0..5 {
            for c in 0..5 {
                assert_eq!(ft[r * 5 + c], [rows[r][c], cols[r][c]], "cell {r},{c}");
            }
        }
        let grid = vec![1.0, f64::NAN, f64::NAN, 2.0];
        let filled = fill_nearest(&grid, &[true, false, false, true], 2, 2).unwrap();
        assert!(filled.iter().all(|v| v.is_finite()));
        assert!(fill_nearest(&grid, &[false; 4], 2, 2).is_err());
    }

    #[test]
    fn coreg_ground_slope_cap() {
        let (ny, nx) = (7, 7);
        let mut g = vec![0.0; ny * nx];
        g[3 * nx + 3] = 10.0;
        let out = enforce_max_slope(g, ny, nx, 0.5, 1.0);
        assert!((out[3 * nx + 3] - 0.5).abs() < 1e-12);
        assert!(out.iter().enumerate().all(|(i, &v)| i == 3 * nx + 3 || v == 0.0));
    }

    #[test]
    fn coreg_ground_errors() {
        assert!(fit_ground(&[], &params()).is_err());
        assert!(fit_ground(&[[0.0, 0.0, 0.0]], &GroundParams { cell_size: 0.0, ..params() }).is_err());
        let e = fit_ground(&[[0.0, 0.0, 0.0]], &GroundParams { min_points_per_cell: 2, ..params() }).unwrap_err();
        assert_eq!(e.to_string(), "no ground cells were observed");
    }

    #[test]
    fn coreg_ground_thinning() {
        let s = sample_sorted(1000, 100, 3);
        assert_eq!(s.len(), 100);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        let pts = cloud(|_, _| 1.0, 20_000);
        let g = fit_ground(&pts, &GroundParams { max_points: Some(5000), ..params() }).unwrap();
        let full = fit_ground(&pts, &params()).unwrap();
        assert_eq!((g.nx, g.ny, g.origin), (full.nx, full.ny, full.origin));
    }
}
