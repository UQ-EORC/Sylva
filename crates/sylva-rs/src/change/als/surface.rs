// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Canopy height, surface and terrain models of two surveys differenced
//! with a level of detection per cell.
//!
//! Each chunk reads the same buffered box from both surveys (the second
//! moved into the first's frame by an [`Alignment`]), optionally thins the
//! denser survey towards the other's pulse density, normalises each survey
//! by a DTM of its own ground returns, and grids both with the same
//! algorithm: a cell's value is its highest return.
//!
//! **Sampling.** The highest of `n` returns is a sample: it falls short of
//! the canopy top by an amount that depends on `n` and on how the canopy's
//! heights are distributed in the cell, and at the edge of a crown one
//! survey may hit the crown where the other hits only the ground. Both are
//! read from the data. Within a cell each pulse contributes one value (its
//! highest return there), and if the canopy did not change, the pulses of
//! the two surveys are samples of one surface: every split of the pooled
//! values into `n_a` and `n_b` is equally likely. The distribution of
//! `max(b) - max(a)` over those splits, which follows exactly from the order
//! statistics of the pooled values (a permutation test on the maxima), is
//! the change the cell would show from sampling alone. Its mean is the
//! sampling bias (negative where the second survey is sparser: its highest
//! return falls shorter of the top) and its central `confidence` interval
//! bounds the change expected without any.
//!
//! **Other errors.** For heights above ground, each survey's DTM under the
//! cell errs by `noise / sqrt(n_ground)` from the ground returns within
//! `dtm_resolution`, plus `interpolation_error` times the distance to the
//! nearest ground return. The alignment errs horizontally, which moves a
//! surface by its gradient times the offset, and for DSMs and DTMs
//! vertically (a CHM is normalised by each survey's own ground, which
//! cancels a vertical offset). These are independent of the sampling: the
//! interval is widened to a half-width of `sqrt(h² + (z s)²)`, `h` its
//! sampling half-width, `s` their combined standard deviation and `z` the
//! normal quantile of the confidence.
//!
//! A cell's change is a gain above the interval, a loss below it, and below
//! detection within it; cells with fewer than `min_returns` pulses in
//! either survey are no data. A DTM is compared the same way with the
//! interpolation and alignment errors alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::als::{catalog_grid, chunk_bounds, in_core, mosaic, output_path, plan, run, workers_for_estimates, write_like, Catalog, Chunk, Layout, BYTES_PER_POINT};
use crate::error::{Error, Result};
use crate::ground::{self, GROUND_CLASS};
use crate::geo::interpolate::{self, GridMethod, GridParams};
use crate::raster::Raster;
use crate::{Point, PointCloud};

use super::{classes, est_other, first_returns, harmonise_pair, median, thin_mask, without_noise, z_of, Alignment, Grid2, Harmonise};

/// Which surface is compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Highest return per cell above each survey's own ground.
    Chm,
    /// Highest return per cell (its elevation).
    Dsm,
    /// Ground surface interpolated from each survey's ground returns.
    Dtm,
}

impl Surface {
    pub fn parse(s: &str) -> Result<Surface> {
        match s {
            "chm" => Ok(Surface::Chm),
            "dsm" => Ok(Surface::Dsm),
            "dtm" => Ok(Surface::Dtm),
            other => Err(Error::invalid(format!("surface must be 'chm', 'dsm' or 'dtm', got {other:?}"))),
        }
    }
}

/// How each survey's DTM is made from its ground returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtmKind {
    /// At each cell centre, a plane fitted by least squares to the ground
    /// returns within a radius (1.5 cells, doubled until it holds six, at
    /// most [`PLANE_RADIUS`]; no value beyond): unbiased by the density of
    /// ground returns, with the standard error of the plane's height there,
    /// and the same whatever the chunks, since each cell depends only on
    /// the returns near it.
    Plane,
    /// Linear interpolation on the Delaunay triangulation (cells outside it
    /// from the nearest): unbiased by the density of ground returns.
    Tin,
    /// The lowest ground return per cell (as `als.dtm`), gaps filled; lower
    /// where there are more returns.
    Lowest,
}

/// Settings of [`surface_change`].
#[derive(Debug, Clone)]
pub struct SurfaceParams {
    pub surface: Surface,
    pub resolution: f64,
    pub dtm_resolution: f64,
    pub dtm: DtmKind,
    /// Grid first returns only (both surveys).
    pub first_returns: bool,
    /// lidR's `p2r(subcircle)`: each return replaced by eight points on a
    /// circle of this radius (m) around it; 0 for the returns themselves.
    pub subcircle: f64,
    /// Fewest returns in a cell of each survey.
    pub min_returns: usize,
    /// Return noise of each survey (m, one standard deviation).
    pub noise_a: f64,
    pub noise_b: f64,
    /// Growth of the DTM's error with distance to the nearest ground
    /// return (m per m).
    pub interpolation_error: f64,
    pub confidence: f64,
    pub harmonise: Option<Harmonise>,
    /// Alignment uncertainty (m) used when no alignment is given.
    pub horizontal_sigma: f64,
    pub vertical_sigma: f64,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        SurfaceParams {
            surface: Surface::Chm,
            resolution: 1.0,
            dtm_resolution: 1.0,
            dtm: DtmKind::Plane,
            first_returns: false,
            subcircle: 0.0,
            min_returns: 2,
            noise_a: 0.05,
            noise_b: 0.05,
            interpolation_error: 0.02,
            confidence: 0.95,
            harmonise: None,
            horizontal_sigma: 0.0,
            vertical_sigma: 0.0,
        }
    }
}

impl SurfaceParams {
    pub fn check(&self) -> Result<()> {
        for (name, v) in [("resolution", self.resolution), ("dtm_resolution", self.dtm_resolution)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
            }
        }
        for (name, v) in [("subcircle", self.subcircle), ("noise_a", self.noise_a), ("noise_b", self.noise_b), ("interpolation_error", self.interpolation_error), ("horizontal_sigma", self.horizontal_sigma), ("vertical_sigma", self.vertical_sigma)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
            }
        }
        if self.min_returns < 1 {
            return Err(Error::invalid("min_returns must be at least 1"));
        }
        z_of(self.confidence)?;
        if let Some(h) = &self.harmonise {
            h.check()?;
        }
        Ok(())
    }
}

/// What a survey's returns look like, for comparing sensors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EpochStats {
    /// Returns and first returns (pulses) in the area compared.
    pub n_returns: u64,
    pub n_first: u64,
    /// First returns that were a pulse's only return.
    pub n_single: u64,
    /// Sum of `number_of_returns` over first returns.
    pub sum_returns_per_pulse: f64,
    /// Sum of `|scan_angle|` (degrees) over first returns that have one.
    pub sum_abs_scan_angle: f64,
    pub n_scan_angle: u64,
    /// Largest `number_of_returns`.
    pub max_returns: u32,
    /// Area (m²) of the cells holding returns.
    pub area: f64,
}

impl EpochStats {
    fn add(&mut self, o: &EpochStats) {
        self.n_returns += o.n_returns;
        self.n_first += o.n_first;
        self.n_single += o.n_single;
        self.sum_returns_per_pulse += o.sum_returns_per_pulse;
        self.sum_abs_scan_angle += o.sum_abs_scan_angle;
        self.n_scan_angle += o.n_scan_angle;
        self.max_returns = self.max_returns.max(o.max_returns);
    }

    fn of(cloud: &PointCloud, buffer: &[bool]) -> EpochStats {
        let first = first_returns(cloud);
        let nr = cloud.attr("number_of_returns");
        let sa = cloud.attr("scan_angle");
        let mut s = EpochStats::default();
        for i in 0..cloud.len() {
            if buffer[i] {
                continue;
            }
            s.n_returns += 1;
            let k = nr.map_or(1.0, |a| a.get_f64(i));
            s.max_returns = s.max_returns.max(k as u32);
            if first[i] {
                s.n_first += 1;
                s.sum_returns_per_pulse += k;
                if k <= 1.0 {
                    s.n_single += 1;
                }
                if let Some(a) = sa {
                    s.sum_abs_scan_angle += a.get_f64(i).abs();
                    s.n_scan_angle += 1;
                }
            }
        }
        s
    }

    /// Pulses per m².
    pub fn pulse_density(&self) -> f64 {
        if self.area > 0.0 { self.n_first as f64 / self.area } else { f64::NAN }
    }

    pub fn returns_per_pulse(&self) -> f64 {
        if self.n_first > 0 { self.sum_returns_per_pulse / self.n_first as f64 } else { f64::NAN }
    }

    pub fn mean_abs_scan_angle(&self) -> f64 {
        if self.n_scan_angle > 0 { self.sum_abs_scan_angle / self.n_scan_angle as f64 } else { f64::NAN }
    }

    pub fn single_share(&self) -> f64 {
        if self.n_first > 0 { self.n_single as f64 / self.n_first as f64 } else { f64::NAN }
    }
}

/// Names of the classes of a change cell, by code.
pub const CLASSES: [&str; 4] = ["no_data", "below_detection", "gain", "loss"];

/// Result of [`surface_change`].
#[derive(Debug, Clone)]
pub struct SurfaceChange {
    /// The two surfaces on the first survey's catalogue grid.
    pub a: Raster,
    pub b: Raster,
    /// `b - a` where the cell could be assessed (NaN elsewhere).
    pub difference: Raster,
    /// Change expected from sampling alone (the mean of the permutation
    /// distribution), and the interval `[lower, upper]` a change must leave
    /// to be significant; `lod` is its half-width.
    pub bias: Raster,
    pub lower: Raster,
    pub upper: Raster,
    pub lod: Raster,
    /// Standard deviation of the change without any: sampling, DTM and
    /// alignment.
    pub sigma: Raster,
    /// Codes of [`CLASSES`], row-major.
    pub classes: Vec<u8>,
    /// Standard deviation of each survey's value: the spread of its highest
    /// pulse under the permutation, and its DTM's error.
    pub sigma_a: Raster,
    pub sigma_b: Raster,
    /// Pulses with a return in the cell, and first returns per m².
    pub pulses_a: Raster,
    pub pulses_b: Raster,
    pub density_a: Raster,
    pub density_b: Raster,
    /// Volumes (m³) raised and lowered over the significant cells, their
    /// area (m²) and the area assessed.
    pub volume_gained: f64,
    pub volume_lost: f64,
    pub area_changed: f64,
    pub area_compared: f64,
    /// The sensors as they were, and as compared (after harmonisation).
    pub raw_a: EpochStats,
    pub raw_b: EpochStats,
    pub stats_a: EpochStats,
    pub stats_b: EpochStats,
    /// Median of `bias` over cells that are canopy (above 2 m) in both.
    pub median_bias: f64,
    /// Sensor differences that bias the comparison, in words.
    pub notes: Vec<String>,
}

/// `ln n!` for `n` up to a bound, grown on demand.
pub struct LnFactorial(Vec<f64>);

impl Default for LnFactorial {
    fn default() -> Self {
        Self::new()
    }
}

impl LnFactorial {
    pub fn new() -> Self {
        LnFactorial(vec![0.0])
    }

    fn get(&mut self, n: usize) -> f64 {
        while self.0.len() <= n {
            let k = self.0.len();
            let last = self.0[k - 1];
            self.0.push(last + (k as f64).ln());
        }
        self.0[n]
    }

    fn ln_choose(&mut self, n: usize, k: usize) -> f64 {
        self.get(n) - self.get(k) - self.get(n - k)
    }
}

/// The change of a cell's highest value that sampling alone would give.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Null {
    /// Mean and standard deviation of `max(b) - max(a)`.
    pub mean: f64,
    pub sd: f64,
    /// Its `alpha / 2` and `1 - alpha / 2` quantiles.
    pub lower: f64,
    pub upper: f64,
    /// Standard deviations of `max(a)` and `max(b)` alone.
    pub sd_a: f64,
    pub sd_b: f64,
}

/// Distribution of `max(B) - max(A)` when the pooled values (sorted
/// ascending) are split at random into `na` and `nb`. If the overall
/// highest value `v_N` falls in `A` (probability `na / N`), `B`'s highest is
/// `v_j` with probability `C(j - 1, nb - 1) / C(N - 1, nb)`, and the other way
/// round; `max(A)` alone is `v_j` with probability `C(j - 1, na - 1) / C(N, na)`.
pub fn null_of_maxima(v: &[f64], na: usize, nb: usize, alpha: f64, lf: &mut LnFactorial) -> Null {
    let n = v.len();
    debug_assert_eq!(n, na + nb);
    let mut atoms: Vec<(f64, f64)> = Vec::with_capacity(n);
    let top = v[n - 1];
    // v_N in A: B's maximum among the other N - 1 values.
    let pa = na as f64 / n as f64;
    let pb = nb as f64 / n as f64;
    let lc_b = lf.ln_choose(n - 1, nb);
    for j in nb..n {
        atoms.push((v[j - 1] - top, pa * (lf.ln_choose(j - 1, nb - 1) - lc_b).exp()));
    }
    let lc_a = lf.ln_choose(n - 1, na);
    for j in na..n {
        atoms.push((top - v[j - 1], pb * (lf.ln_choose(j - 1, na - 1) - lc_a).exp()));
    }
    let total: f64 = atoms.iter().map(|a| a.1).sum();
    let mean = atoms.iter().map(|a| a.0 * a.1).sum::<f64>() / total;
    let var = atoms.iter().map(|a| (a.0 - mean).powi(2) * a.1).sum::<f64>() / total;
    atoms.sort_by(|x, y| x.0.total_cmp(&y.0));
    let quantile = |q: f64| -> f64 {
        let mut c = 0.0;
        for a in &atoms {
            c += a.1 / total;
            if c >= q - 1e-12 {
                return a.0;
            }
        }
        atoms[atoms.len() - 1].0
    };
    let mut sd_max = |k: usize| -> f64 {
        let lc = lf.ln_choose(n, k);
        let (mut m1, mut m2, mut w) = (0.0, 0.0, 0.0);
        for j in k..=n {
            let p = (lf.ln_choose(j - 1, k - 1) - lc).exp();
            m1 += p * v[j - 1];
            m2 += p * v[j - 1] * v[j - 1];
            w += p;
        }
        let m = m1 / w;
        (m2 / w - m * m).max(0.0).sqrt()
    };
    let (sd_a, sd_b) = (sd_max(na), sd_max(nb));
    Null { mean, sd: var.max(0.0).sqrt(), lower: quantile(alpha / 2.0), upper: quantile(1.0 - alpha / 2.0), sd_a, sd_b }
}

/// Per-cell values of one survey over a chunk window.
struct EpochCells {
    /// Highest return of each pulse in each cell.
    units: Vec<Vec<f64>>,
    /// First returns per m².
    first: Vec<f64>,
    /// Standard deviation of the survey's DTM under each cell.
    dtm_sd: Vec<f64>,
    /// The DTM itself at the change resolution (for the DTM surface).
    dtm: Vec<f64>,
}

/// Largest radius (m) of the ground planes of [`DtmKind::Plane`].
pub const PLANE_RADIUS: f64 = 16.0;

/// A survey's DTM over the chunk's buffered box on `grid`, and its standard
/// deviation per cell: for [`DtmKind::Plane`] the plane's standard error
/// (from its residuals, at least `noise`), otherwise `noise / sqrt(n)` with
/// `n` the ground returns within a cell's width; both plus
/// `interpolation_error` times the distance to the nearest ground return.
/// `ground` must be in canonical order ([`super::canonical`]); None with
/// fewer than 3 ground returns.
pub(crate) fn chunk_dtm(ground: &[Point], grid: &Raster, outer: &[f64; 4], kind: DtmKind, noise: f64, interpolation_error: f64, reach: f64) -> Result<Option<(Raster, Raster)>> {
    if ground.len() < 3 {
        return Ok(None);
    }
    let b = chunk_bounds(grid, outer);
    let res = grid.resolution;
    let gg = Grid2::new(ground, res.max(1.0));
    let mut sd = Raster::from_points(std::iter::empty(), std::iter::empty(), res, crate::raster::Reducer::Min, Some(b), f64::NAN)?;
    let value = match kind {
        DtmKind::Lowest => Some(ground::make_dtm(ground, res, Some(b))?),
        DtmKind::Tin => {
            let z: Vec<f64> = ground.iter().map(|q| q[2]).collect();
            let mut r = interpolate::grid(ground, &z, res, Some(b), &GridParams { method: GridMethod::Tin, power: 2.0, k: 12, max_distance: None })?;
            if r.data.iter().any(|v| v.is_nan()) {
                r.fill_nearest();
            }
            Some(r)
        }
        DtmKind::Plane => None,
    };
    let mut plane = sd.clone();
    let r0 = (1.5 * res).max(1.0);
    for row in 0..sd.nrows {
        for col in 0..sd.ncols {
            let (cx, cy) = sd.cell_center(row, col);
            let d = gg.nearest(ground, cx, cy, reach);
            if !d.is_finite() {
                continue;
            }
            let model = (interpolation_error * d).powi(2);
            let k = row * sd.ncols + col;
            match kind {
                DtmKind::Plane => {
                    let mut r = r0;
                    while r <= reach {
                        let idx = gg.within(ground, cx, cy, r);
                        if idx.len() >= 6 {
                            if let Some(pl) = super::fit_plane(ground, &idx, cx, cy) {
                                plane.data[k] = pl.h;
                                sd.data[k] = (pl.rms.max(noise).powi(2) * pl.var_h + model).sqrt();
                                break;
                            }
                        }
                        r *= 2.0;
                    }
                }
                _ => {
                    let n = gg.within(ground, cx, cy, res).len().max(1) as f64;
                    sd.data[k] = (noise * noise / n + model).sqrt();
                }
            }
        }
    }
    let value = value.unwrap_or(plane);
    if value.data.iter().all(|v| v.is_nan()) {
        return Ok(None);
    }
    Ok(Some((value, sd)))
}

/// One survey's cells over the chunk window `(xmin, ymin, nrows, ncols)`.
#[allow(clippy::too_many_arguments)]
fn epoch_cells(cloud: &PointCloud, heights: &[f64], dtm_sd: &Raster, dtm_fine: Option<&(Raster, Raster)>, win: (f64, f64, usize, usize), p: &SurfaceParams) -> EpochCells {
    let (x0, y0, nr, nc) = win;
    let res = p.resolution;
    let n = nr * nc;
    let first = first_returns(cloud);
    let keys = super::pulse_keys(cloud);
    let mut firsts = vec![0usize; n];
    // One height per cell per pulse, the highest that pulse returned there.
    // A cell's height is then compared across surveys pulse by pulse rather
    // than point by point, so a cell that happened to be hit by more pulses
    // in one survey does not look taller for it.
    let mut best: HashMap<(usize, u64), f64> = HashMap::new();
    // Which cell of this chunk's window a coordinate falls in, if any; the
    // window is a window on a larger catalogue, so points outside it are not
    // this chunk's business.
    let cell_of = |x: f64, y: f64| -> Option<usize> {
        let (r, c) = (((y - y0) / res).floor(), ((x - x0) / res).floor());
        if r >= 0.0 && c >= 0.0 && (r as usize) < nr && (c as usize) < nc { Some(r as usize * nc + c as usize) } else { None }
    };
    if p.surface != Surface::Dtm {
        for (i, q) in cloud.xyz.iter().enumerate() {
            if first[i] {
                if let Some(k) = cell_of(q[0], q[1]) {
                    firsts[k] += 1;
                }
            }
            if p.first_returns && !first[i] {
                continue;
            }
            let h = if p.surface == Surface::Dsm { q[2] } else { heights[i] };
            if !h.is_finite() {
                continue;
            }
            let mut put = |k: usize| {
                let e = best.entry((k, keys[i])).or_insert(f64::NEG_INFINITY);
                if h > *e {
                    *e = h;
                }
            };
            // A point can be spread over a small circle before being binned,
            // which fills the gaps between returns on a sparse survey: the
            // point claims the eight cells around it as well as its own.
            if p.subcircle > 0.0 {
                for a in 0..8 {
                    let ang = a as f64 * std::f64::consts::FRAC_PI_4;
                    if let Some(k) = cell_of(q[0] + p.subcircle * ang.cos(), q[1] + p.subcircle * ang.sin()) {
                        put(k);
                    }
                }
            } else if let Some(k) = cell_of(q[0], q[1]) {
                put(k);
            }
        }
    }
    // Gather the per-pulse heights into a sorted list per cell. Sorting makes
    // the later statistics (a median, a percentile) straightforward, and it
    // also means the result does not depend on the order the hash map
    // happened to hand them back, which varies between runs.
    let mut units = vec![Vec::new(); n];
    for ((k, _), h) in best {
        units[k].push(h);
    }
    for u in units.iter_mut() {
        u.sort_by(|a, b| a.total_cmp(b));
    }
    let area = res * res;
    let mut sd = vec![f64::NAN; n];
    let mut dtm = vec![f64::NAN; n];
    for k in 0..n {
        let (r, c) = (k / nc, k % nc);
        let (cx, cy) = (x0 + (c as f64 + 0.5) * res, y0 + (r as f64 + 0.5) * res);
        // How uncertain the ground is under this cell, which later sets how
        // large a height change has to be to be believed. A surface model
        // measured from the top does not stand on the DTM at all, so it
        // carries none of its uncertainty.
        match (p.surface, dtm_fine) {
            (Surface::Dsm, _) => sd[k] = 0.0,
            (Surface::Dtm, Some((v, s))) => {
                dtm[k] = v.data[k];
                sd[k] = s.data[k];
            }
            _ => {
                let (row, col) = dtm_sd.cell_index(cx, cy);
                if dtm_sd.in_bounds(row, col) {
                    sd[k] = dtm_sd.get(row as usize, col as usize);
                }
            }
        }
    }
    EpochCells { units, first: firsts.iter().map(|&f| f as f64 / area).collect(), dtm_sd: sd, dtm }
}

/// Heights above a chunk's DTM, sampled bilinearly between cell centres as
/// [`Raster::sample`] does, but with the interpolation weights taken from
/// the position on the whole catalogue's grid (corner `origin`), so that a
/// point's height does not depend on where its chunk's raster starts.
pub(crate) fn heights(cloud: &PointCloud, dtm: &Raster, origin: (f64, f64)) -> Vec<f64> {
    let res = dtm.resolution;
    let (c_off, r_off) = (((dtm.xmin - origin.0) / res).round(), ((dtm.ymin - origin.1) / res).round());
    let (nc, nr) = (dtm.ncols as f64, dtm.nrows as f64);
    cloud
        .xyz
        .iter()
        .map(|q| {
            let fc = ((q[0] - origin.0) / res - 0.5 - c_off).clamp(0.0, nc - 1.0);
            let fr = ((q[1] - origin.1) / res - 0.5 - r_off).clamp(0.0, nr - 1.0);
            let (c0, r0) = (fc.floor() as usize, fr.floor() as usize);
            let (c1, r1) = ((c0 + 1).min(dtm.ncols - 1), (r0 + 1).min(dtm.nrows - 1));
            let (tx, ty) = (fc - c0 as f64, fr - r0 as f64);
            let z = dtm.get(r0, c0) * (1.0 - tx) * (1.0 - ty) + dtm.get(r0, c1) * tx * (1.0 - ty) + dtm.get(r1, c0) * (1.0 - tx) * ty + dtm.get(r1, c1) * tx * ty;
            q[2] - z
        })
        .collect()
}

/// Ground returns in canonical order.
pub(crate) fn ground_of(cloud: &PointCloud) -> Vec<Point> {
    let cls = classes(cloud);
    super::canonical((0..cloud.len()).filter(|&i| cls[i] == GROUND_CLASS).map(|i| cloud.xyz[i]).collect())
}

/// Gradient components of a raster by central differences (one-sided at
/// NaN neighbours and edges, 0 where both are missing).
pub fn gradient(r: &Raster) -> (Vec<f64>, Vec<f64>) {
    let (nr, nc, res) = (r.nrows, r.ncols, r.resolution);
    let mut gx = vec![0.0; nr * nc];
    let mut gy = vec![0.0; nr * nc];
    let val = |row: i64, col: i64| -> f64 { if row < 0 || col < 0 || row >= nr as i64 || col >= nc as i64 { f64::NAN } else { r.get(row as usize, col as usize) } };
    let d = |lo: f64, mid: f64, hi: f64| -> f64 {
        match (lo.is_finite(), hi.is_finite()) {
            (true, true) => (hi - lo) / (2.0 * res),
            (true, false) if mid.is_finite() => (mid - lo) / res,
            (false, true) if mid.is_finite() => (hi - mid) / res,
            _ => 0.0,
        }
    };
    for row in 0..nr as i64 {
        for col in 0..nc as i64 {
            let m = val(row, col);
            let k = row as usize * nc + col as usize;
            gx[k] = d(val(row, col - 1), m, val(row, col + 1));
            gy[k] = d(val(row - 1, col), m, val(row + 1, col));
        }
    }
    (gx, gy)
}

/// Sensor differences that bias CHM change, in words.
pub fn sensor_notes(a: &EpochStats, b: &EpochStats, raw_a: &EpochStats, raw_b: &EpochStats, median_bias: f64, harmonised: bool) -> Vec<String> {
    let mut notes = Vec::new();
    let (da, db) = (raw_a.pulse_density(), raw_b.pulse_density());
    if da.is_finite() && db.is_finite() && da.min(db) > 0.0 && da.max(db) / da.min(db) > 1.25 {
        let (dense, ratio) = if db > da { ("b", db / da) } else { ("a", da / db) };
        let mut s = format!(
            "pulse density differs: {da:.1} pulses/m² in survey a and {db:.1} in survey b (survey {dense} {ratio:.1} times denser). The highest return in a cell lies nearer the true canopy top where returns are denser, so CHM change is biased towards the denser survey"
        );
        if median_bias.is_finite() {
            s += &format!(" (by {median_bias:+.2} m in the median canopy cell, as the pulses of the two surveys sample it)");
        }
        if harmonised {
            s += &format!("; after harmonisation the densities compared are {:.1} and {:.1} pulses/m²", a.pulse_density(), b.pulse_density());
        } else {
            s += "; harmonise=True thins the denser survey to the other's density";
        }
        notes.push(s + ".");
    }
    let (ra, rb) = (raw_a.returns_per_pulse(), raw_b.returns_per_pulse());
    if ra.is_finite() && rb.is_finite() && (ra - rb).abs() > 0.2 * ra.min(rb) {
        notes.push(format!(
            "returns per pulse differ ({ra:.2} in survey a, {rb:.2} in survey b; single-return pulses {:.0} % and {:.0} %): the sensors differ in footprint, sensitivity or range resolution, which changes where in a crown the first return is triggered and how deep pulses reach. Compare first returns only (first_returns=True) and read change near the canopy surface with care.",
            100.0 * raw_a.single_share(),
            100.0 * raw_b.single_share()
        ));
    }
    if raw_a.max_returns != raw_b.max_returns && raw_a.max_returns > 0 && raw_b.max_returns > 0 {
        notes.push(format!("the surveys record up to {} and {} returns per pulse; below-canopy returns and ground coverage differ with it.", raw_a.max_returns, raw_b.max_returns));
    }
    let (sa, sb) = (raw_a.mean_abs_scan_angle(), raw_b.mean_abs_scan_angle());
    if sa.is_finite() && sb.is_finite() && (sa - sb).abs() > 5.0 {
        notes.push(format!(
            "scan angles differ (mean {sa:.1} and {sb:.1} degrees from nadir): oblique pulses see crown sides and are occluded differently, which changes CHMs at crown edges and in small gaps."
        ));
    }
    notes
}

/// Compare a surface of two surveys over the catalogue of the first; see
/// the module documentation. `alignment` moves the second survey into the
/// first's frame; without it, `horizontal_sigma` and `vertical_sigma` are
/// the alignment uncertainty.
#[allow(clippy::needless_range_loop)]
pub fn surface_change(cat_a: &Catalog, cat_b: &Catalog, alignment: Option<&Alignment>, p: &SurfaceParams, layout: Layout, buffer: f64, workers: usize) -> Result<SurfaceChange> {
    p.check()?;
    cat_b.check_usable()?;
    if !(buffer.is_finite() && buffer >= 0.0) {
        return Err(Error::invalid(format!("buffer must be a non-negative number of metres, got {buffer}")));
    }
    let res = p.resolution;
    let alpha = 1.0 - p.confidence;
    let grid = catalog_grid(cat_a, res)?;
    let dgrid = catalog_grid(cat_a, p.dtm_resolution)?;
    // Whole cells, and the ground planes of the DTM cells under them.
    let mut buffer = buffer.max(2.0 * res.max(p.dtm_resolution)).max(PLANE_RADIUS + res + 2.0 * p.dtm_resolution);
    if let Some(h) = &p.harmonise {
        buffer = buffer.max(h.cell + 2.0 * res);
    }
    let grow = alignment.map_or(0.0, |a| a.max_horizontal() + 0.5);
    let chunks = plan(cat_a, layout, buffer)?;
    let est: Vec<u64> = chunks.iter().map(|c| c.est_points + est_other(cat_b, c, grow)).collect();
    let w = workers_for_estimates(&est, workers, BYTES_PER_POINT)?;
    const LAYERS: usize = 14;
    let parts = run(cat_a, &chunks, w, "surface change", |chunk, data| {
        let a = without_noise(data.cloud, data.buffer);
        let (braw, _) = super::read_other(cat_b, chunk, grow, None)?;
        let bbuf = vec![true; braw.len()];
        let b = without_noise(braw, bbuf);
        let raw_a = EpochStats::of(&a.0, &a.1);
        let raw_b = EpochStats::of(&b.0, &core_flags(&b.0, chunk, alignment));
        // Thin before moving the second survey, so that the pulses kept are
        // those harmonise_catalog keeps.
        let (a, b) = match &p.harmonise {
            Some(h) => harmonise_pair(a, b, h),
            None => (a, b),
        };
        let b = shift_crop(b.0, chunk, alignment);
        let (stats_a, stats_b) = (EpochStats::of(&a.0, &a.1), EpochStats::of(&b.0, &b.1));
        let (ga, gb) = (ground_of(&a.0), ground_of(&b.0));
        let reach = PLANE_RADIUS;
        let (Some(da), Some(db)) = (chunk_dtm(&ga, &dgrid, &chunk.outer, p.dtm, p.noise_a, p.interpolation_error, reach)?, chunk_dtm(&gb, &dgrid, &chunk.outer, p.dtm, p.noise_b, p.interpolation_error, reach)?) else {
            return Ok(None);
        };
        let bnd = chunk_bounds(&grid, &chunk.outer);
        let nc = ((bnd.2 - bnd.0) / res).floor() as usize + 1;
        let nr = ((bnd.3 - bnd.1) / res).floor() as usize + 1;
        let win = (bnd.0, bnd.1, nr, nc);
        let fine = |g: &[Point], noise: f64| -> Result<Option<(Raster, Raster)>> {
            if p.surface != Surface::Dtm {
                return Ok(None);
            }
            chunk_dtm(g, &grid, &chunk.outer, p.dtm, noise, p.interpolation_error, reach)
        };
        let (fa, fb) = (fine(&ga, p.noise_a)?, fine(&gb, p.noise_b)?);
        let origin = (dgrid.xmin, dgrid.ymin);
        let ca = epoch_cells(&a.0, &heights(&a.0, &da.0, origin), &da.1, fa.as_ref(), win, p);
        let cb = epoch_cells(&b.0, &heights(&b.0, &db.0, origin), &db.1, fb.as_ref(), win, p);
        let n = nr * nc;
        let mut layers = vec![vec![f64::NAN; n]; LAYERS];
        let mut lf = LnFactorial::new();
        for k in 0..n {
            layers[10][k] = ca.first[k];
            layers[11][k] = cb.first[k];
            layers[12][k] = ca.dtm_sd[k];
            layers[13][k] = cb.dtm_sd[k];
            if p.surface == Surface::Dtm {
                layers[0][k] = ca.dtm[k];
                layers[1][k] = cb.dtm[k];
                for l in [2, 3, 4, 5, 6, 7] {
                    layers[l][k] = 0.0;
                }
                continue;
            }
            let (ua, ub) = (&ca.units[k], &cb.units[k]);
            layers[8][k] = ua.len() as f64;
            layers[9][k] = ub.len() as f64;
            if let Some(v) = ua.last() {
                layers[0][k] = *v;
            }
            if let Some(v) = ub.last() {
                layers[1][k] = *v;
            }
            if ua.is_empty() || ub.is_empty() {
                continue;
            }
            let mut pooled: Vec<f64> = ua.iter().chain(ub.iter()).copied().collect();
            pooled.sort_by(|x, y| x.total_cmp(y));
            let nl = null_of_maxima(&pooled, ua.len(), ub.len(), alpha, &mut lf);
            for (l, v) in [(2, nl.mean), (3, nl.sd), (4, nl.lower), (5, nl.upper), (6, nl.sd_a), (7, nl.sd_b)] {
                layers[l][k] = v;
            }
        }
        let rasters: Vec<Raster> = layers.into_iter().map(|v| Raster { data: v, nrows: nr, ncols: nc, xmin: bnd.0, ymin: bnd.1, resolution: res }).collect();
        Ok(Some((rasters, chunk.core, [raw_a, raw_b, stats_a, stats_b])))
    })?;
    let parts: Vec<_> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() {
        return Err(Error::invalid("no chunk has 3 ground returns (classification 2) in both surveys; classify ground first (als.classify_ground)"));
    }
    let mut stats = [EpochStats::default(); 4];
    for (_, _, s) in &parts {
        for k in 0..4 {
            stats[k].add(&s[k]);
        }
    }
    let mut m: Vec<Raster> = Vec::with_capacity(LAYERS);
    for k in 0..LAYERS {
        m.push(mosaic(&grid, &parts.iter().map(|(r, core, _)| (r[k].clone(), *core)).collect::<Vec<_>>())?);
    }
    let cell_area = res * res;
    for (k, first) in [(0usize, &m[10]), (1, &m[11])] {
        let cells = first.data.iter().filter(|v| v.is_finite() && **v > 0.0).count() as f64;
        stats[k].area = cells * cell_area;
        stats[k + 2].area = cells * cell_area;
    }
    let (ra, rb) = (&m[0], &m[1]);
    let mean = Raster { data: ra.data.iter().zip(&rb.data).map(|(x, y)| if x.is_finite() && y.is_finite() { 0.5 * (x + y) } else if x.is_finite() { *x } else { *y }).collect(), ..ra.clone() };
    let (gx, gy) = gradient(&mean);
    let z = z_of(p.confidence)?;
    let n = grid.nrows * grid.ncols;
    let mut out = vec![vec![f64::NAN; n]; 8];
    let mut cls = vec![0u8; n];
    let (mut gained, mut lost, mut n_changed, mut n_compared) = (0.0, 0.0, 0usize, 0usize);
    let min_n = p.min_returns.max(1) as f64;
    for k in 0..n {
        let (row, col) = (k / grid.ncols, k % grid.ncols);
        let (x, y) = grid.cell_center(row, col);
        let s_al = match alignment {
            Some(al) => al.sigma_at(x, y),
            None => [p.horizontal_sigma, p.horizontal_sigma, p.vertical_sigma],
        };
        let d = rb.data[k] - ra.data[k];
        let (dsa, dsb) = (if p.surface == Surface::Dsm { 0.0 } else { m[12].data[k] }, if p.surface == Surface::Dsm { 0.0 } else { m[13].data[k] });
        let mut extra = dsa * dsa + dsb * dsb + (s_al[0] * gx[k]).powi(2) + (s_al[1] * gy[k]).powi(2);
        if p.surface != Surface::Chm {
            extra += s_al[2] * s_al[2];
        }
        let short = p.surface != Surface::Dtm && (m[8].data[k] < min_n || m[9].data[k] < min_n);
        let (lo, hi, nmean, nsd) = (m[4].data[k], m[5].data[k], m[2].data[k], m[3].data[k]);
        if !d.is_finite() || !extra.is_finite() || short || !lo.is_finite() || !hi.is_finite() {
            continue;
        }
        let centre = 0.5 * (lo + hi);
        let half = (0.5 * (hi - lo)).hypot(z * extra.sqrt());
        out[0][k] = d;
        out[1][k] = nmean;
        out[2][k] = centre - half;
        out[3][k] = centre + half;
        out[4][k] = half;
        out[5][k] = (nsd * nsd + extra).sqrt();
        out[6][k] = m[6].data[k].hypot(dsa);
        out[7][k] = m[7].data[k].hypot(dsb);
        n_compared += 1;
        cls[k] = if d > centre + half {
            gained += d * cell_area;
            n_changed += 1;
            2
        } else if d < centre - half {
            lost -= d * cell_area;
            n_changed += 1;
            3
        } else {
            1
        };
    }
    let r = |v: Vec<f64>| Raster { data: v, ..grid.clone() };
    let mut canopy: Vec<f64> = (0..n).filter(|&k| ra.data[k] > 2.0 && rb.data[k] > 2.0 && out[1][k].is_finite()).map(|k| out[1][k]).collect();
    let median_bias = if p.surface == Surface::Dtm { f64::NAN } else { median(&mut canopy) };
    let notes = if p.surface == Surface::Dtm { Vec::new() } else { sensor_notes(&stats[2], &stats[3], &stats[0], &stats[1], median_bias, p.harmonise.is_some()) };
    let per_m2 = |v: &Raster| r(v.data.iter().map(|x| x / cell_area).collect());
    let mut it = out.into_iter();
    let mut next = || r(it.next().expect("a layer"));
    Ok(SurfaceChange {
        difference: next(),
        bias: next(),
        lower: next(),
        upper: next(),
        lod: next(),
        sigma: next(),
        sigma_a: next(),
        sigma_b: next(),
        classes: cls,
        pulses_a: per_m2(&m[8]),
        pulses_b: per_m2(&m[9]),
        density_a: m[10].clone(),
        density_b: m[11].clone(),
        a: m[0].clone(),
        b: m[1].clone(),
        volume_gained: gained,
        volume_lost: lost,
        area_changed: n_changed as f64 * cell_area,
        area_compared: n_compared as f64 * cell_area,
        raw_a: stats[0],
        raw_b: stats[1],
        stats_a: stats[2],
        stats_b: stats[3],
        median_bias,
        notes,
    })
}

/// True for the points of the second survey that do not fall in the
/// chunk's core once moved into the first survey's frame.
pub(crate) fn core_flags(cloud: &PointCloud, chunk: &Chunk, alignment: Option<&Alignment>) -> Vec<bool> {
    cloud
        .xyz
        .iter()
        .map(|q| {
            let o = alignment.map_or([0.0; 3], |al| al.offset_at(q[0], q[1]));
            !in_core(&chunk.core, q[0] - o[0], q[1] - o[1])
        })
        .collect()
}

/// Move the second survey's chunk into the first survey's frame and keep
/// what falls in the buffered box.
pub(crate) fn shift_crop(mut cloud: PointCloud, chunk: &Chunk, alignment: Option<&Alignment>) -> (PointCloud, Vec<bool>) {
    if let Some(al) = alignment {
        for q in cloud.xyz.iter_mut() {
            let o = al.offset_at(q[0], q[1]);
            q[0] -= o[0];
            q[1] -= o[1];
            q[2] -= o[2];
        }
    }
    let o = chunk.outer;
    let keep: Vec<usize> = (0..cloud.len()).filter(|&i| {
        let q = cloud.xyz[i];
        q[0] >= o[0] && q[0] <= o[2] && q[1] >= o[1] && q[1] <= o[3]
    }).collect();
    let cloud = if keep.len() == cloud.len() { cloud } else { cloud.take(&keep) };
    let buffer = cloud.xyz.iter().map(|q| !in_core(&chunk.core, q[0], q[1])).collect();
    (cloud, buffer)
}

/// Write `cat_self` thinned towards the pulse density of `cat_other`, one
/// file per chunk (its core points), as [`surface_change`] with
/// `harmonise` thins it. Returns the files written.
pub fn harmonise_catalog(cat_self: &Catalog, cat_other: &Catalog, out_dir: &Path, h: &Harmonise, format: Option<&str>, layout: Layout, workers: usize) -> Result<Vec<PathBuf>> {
    h.check()?;
    cat_other.check_usable()?;
    let chunks = plan(cat_self, layout, h.cell + 1.0)?;
    let ext_of = |chunk: &Chunk| -> Result<String> {
        if let Some(f) = format {
            let f = f.trim_start_matches('.').to_ascii_lowercase();
            if f != "las" && f != "laz" {
                return Err(Error::invalid(format!("format must be 'las' or 'laz', got {f:?}")));
            }
            return Ok(f);
        }
        let t = &cat_self.tiles[chunk.own.unwrap_or(chunk.files[0])];
        Ok(t.path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).filter(|e| e == "las" || e == "laz").unwrap_or_else(|| "laz".into()))
    };
    std::fs::create_dir_all(out_dir)?;
    for c in &chunks {
        output_path(cat_self, out_dir, &c.name, &ext_of(c)?)?;
    }
    let est: Vec<u64> = chunks.iter().map(|c| c.est_points + est_other(cat_other, c, 0.0)).collect();
    let w = workers_for_estimates(&est, workers, BYTES_PER_POINT)?;
    let written = run(cat_self, &chunks, w, "harmonising density", |chunk, data| {
        let (other, _) = super::read_other(cat_other, chunk, 0.0, None)?;
        let keep = thin_mask(&data.cloud, &other, h);
        let idx: Vec<usize> = (0..data.cloud.len()).filter(|&i| keep[i] && !data.buffer[i]).collect();
        if idx.is_empty() {
            return Ok(None);
        }
        let out = data.cloud.take(&idx);
        let path = output_path(cat_self, out_dir, &chunk.name, &ext_of(chunk)?)?;
        write_like(&out, &path, &cat_self.tiles[chunk.own.unwrap_or(chunk.files[0])])?;
        Ok(Some(path))
    })?;
    Ok(written.into_iter().flatten().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every split of the pooled values, enumerated.
    fn brute(v: &[f64], na: usize) -> Vec<(f64, f64, f64)> {
        let n = v.len();
        let mut out = Vec::new();
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != na {
                continue;
            }
            let ma = (0..n).filter(|&i| mask >> i & 1 == 1).map(|i| v[i]).fold(f64::NEG_INFINITY, f64::max);
            let mb = (0..n).filter(|&i| mask >> i & 1 == 0).map(|i| v[i]).fold(f64::NEG_INFINITY, f64::max);
            out.push((mb - ma, ma, mb));
        }
        out
    }

    #[test]
    fn the_null_of_the_maxima_matches_enumeration() {
        let v = [0.1, 0.3, 0.35, 1.0, 2.5, 2.6, 4.0, 7.0, 7.2, 9.0];
        let mut lf = LnFactorial::new();
        for na in 1..v.len() {
            let nb = v.len() - na;
            let all = brute(&v, na);
            let k = all.len() as f64;
            let mean = all.iter().map(|x| x.0).sum::<f64>() / k;
            let sd = (all.iter().map(|x| (x.0 - mean).powi(2)).sum::<f64>() / k).sqrt();
            let ma = all.iter().map(|x| x.1).sum::<f64>() / k;
            let sda = (all.iter().map(|x| (x.1 - ma).powi(2)).sum::<f64>() / k).sqrt();
            let nl = null_of_maxima(&v, na, nb, 0.1, &mut lf);
            assert!((nl.mean - mean).abs() < 1e-9 && (nl.sd - sd).abs() < 1e-9, "na {na}: {nl:?} vs {mean} {sd}");
            assert!((nl.sd_a - sda).abs() < 1e-9);
            let mut d: Vec<f64> = all.iter().map(|x| x.0).collect();
            d.sort_by(|a, b| a.total_cmp(b));
            let below = d.iter().filter(|&&x| x < nl.lower).count() as f64 / k;
            let above = d.iter().filter(|&&x| x > nl.upper).count() as f64 / k;
            assert!(below <= 0.05 + 1e-12 && above <= 0.05 + 1e-12, "na {na}: {below} {above}");
            assert!(d.iter().filter(|&&x| x <= nl.lower).count() as f64 / k >= 0.05 - 1e-12);
        }
    }

    #[test]
    fn exchangeable_samples_are_rarely_called_change() {
        // Heights of a crown flank (uniform) and ground: both surveys sample
        // one surface, the second more sparsely.
        let mut s = 11u64;
        let mut rnd = || {
            s = super::super::mix(s);
            super::super::unit(s)
        };
        let mut lf = LnFactorial::new();
        let (mut false_calls, mut bias, trials) = (0, 0.0, 4000);
        for _ in 0..trials {
            let draw = |r: f64| if r < 0.3 { 0.02 * r } else { 12.0 + 6.0 * r };
            let a: Vec<f64> = (0..20).map(|_| draw(rnd())).collect();
            let b: Vec<f64> = (0..6).map(|_| draw(rnd())).collect();
            let d = b.iter().cloned().fold(f64::MIN, f64::max) - a.iter().cloned().fold(f64::MIN, f64::max);
            let mut pooled: Vec<f64> = a.iter().chain(&b).copied().collect();
            pooled.sort_by(|x, y| x.total_cmp(y));
            let nl = null_of_maxima(&pooled, 20, 6, 0.05, &mut lf);
            bias += nl.mean;
            if d < nl.lower || d > nl.upper {
                false_calls += 1;
            }
        }
        let rate = false_calls as f64 / trials as f64;
        assert!(rate <= 0.05, "{rate}");
        // The sparser survey's highest return falls shorter of the top.
        assert!(bias / trials as f64 <= -0.3, "{}", bias / trials as f64);
    }

    #[test]
    fn a_felled_crown_is_a_loss() {
        let a: Vec<f64> = (0..25).map(|k| 15.0 + 0.2 * k as f64).collect();
        let b: Vec<f64> = (0..8).map(|k| 0.01 * k as f64).collect();
        let mut pooled: Vec<f64> = a.iter().chain(&b).copied().collect();
        pooled.sort_by(|x, y| x.total_cmp(y));
        let nl = null_of_maxima(&pooled, 25, 8, 0.05, &mut LnFactorial::new());
        assert!(0.07 - 19.8 < nl.lower, "{nl:?}");
    }

    #[test]
    fn gradients_are_central_differences() {
        let mut r = Raster::filled(3, 4, 0.0, 0.0, 0.5, 0.0);
        for row in 0..3 {
            for col in 0..4 {
                r.set(row, col, 2.0 * col as f64 * 0.5 - 1.0 * row as f64 * 0.5);
            }
        }
        r.set(1, 3, f64::NAN);
        let (gx, gy) = gradient(&r);
        assert!((gx[5] - 2.0).abs() < 1e-12 && (gy[5] + 1.0).abs() < 1e-12);
        assert!((gx[6] - 2.0).abs() < 1e-12, "one-sided next to a NaN");
        assert!((gx[0] - 2.0).abs() < 1e-12, "one-sided at the edge");
    }

    #[test]
    fn notes_name_the_sensor_differences() {
        let a = EpochStats { n_returns: 3000, n_first: 1000, n_single: 500, sum_returns_per_pulse: 2000.0, sum_abs_scan_angle: 10_000.0, n_scan_angle: 1000, max_returns: 5, area: 100.0 };
        let b = EpochStats { n_returns: 1200, n_first: 400, n_single: 300, sum_returns_per_pulse: 480.0, sum_abs_scan_angle: 1_600.0, n_scan_angle: 400, max_returns: 3, area: 100.0 };
        let n = sensor_notes(&a, &b, &a, &b, -0.3, false);
        assert_eq!(n.len(), 4, "{n:?}");
        assert!(n[0].contains("survey a 2.5 times denser") && n[0].contains("-0.30 m"));
        assert!(sensor_notes(&a, &a, &a, &a, 0.0, false).is_empty());
    }
}
