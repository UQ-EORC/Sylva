// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Gap-probability profiles and what follows from them.
//!
//! Stateless functions over the arrays a gap profile holds (hits by ring,
//! azimuth sector and height; pulses fired by ring and sector; the same per
//! scan), so that a profile is plain data in every language that binds
//! them. Also the fired-pulse estimates for RIEGL streams that lack the
//! pulses returning nothing, the ground plane of single-scan profiles, and
//! the height profiles of a ray-traced density grid.

use crate::error::{Error, Result};
use crate::numeric::{arange, gradient, histogram, median, nanmean, quantile, searchsorted_left, searchsorted_right};
use crate::raster::Raster;
use crate::shots::Shots;
use crate::Point;

/// Zenith angle (degrees from up) of each direction.
pub fn zenith_deg(direction: &[Point]) -> Vec<f64> {
    direction.iter().map(|d| d[2].clamp(-1.0, 1.0).acos().to_degrees()).collect()
}

// --------------------------------------------------------------- scan pattern

/// The angular grid of a RIEGL scan, as parsed from a RiSCAN project.
#[derive(Debug, Clone, Copy)]
pub struct ScanPattern {
    /// First zenith line (degrees).
    pub theta_start: f64,
    /// Zenith step between lines (degrees).
    pub theta_delta: f64,
    /// Number of zenith lines.
    pub theta_count: usize,
    /// Azimuth steps, one pulse per zenith line each.
    pub phi_count: usize,
}

impl ScanPattern {
    /// Zenith angle of each line (degrees).
    pub fn lines(&self) -> Vec<f64> {
        (0..self.theta_count).map(|i| self.theta_start + self.theta_delta * i as f64).collect()
    }

    /// Bin edges half a step either side of each zenith line.
    pub fn line_edges(&self) -> Vec<f64> {
        let theta = self.lines();
        let half = self.theta_delta / 2.0;
        let mut e: Vec<f64> = theta.iter().map(|t| t - half).collect();
        if let Some(last) = theta.last() {
            e.push(last + half);
        }
        e
    }
}

/// Effective pulses fired along each zenith line: the larger of the nominal
/// `phi_count / shot_stride` and the `quantile` of the per-line shot counts.
pub fn pulses_per_line(zenith: &[f64], pattern: &ScanPattern, quantile_q: f64, shot_stride: usize) -> usize {
    let observed: Vec<f64> = histogram(zenith.iter().copied(), &pattern.line_edges()).into_iter().map(|c| c as f64).collect();
    let nominal = pattern.phi_count as f64 / shot_stride.max(1) as f64;
    nominal.max(quantile(&observed, quantile_q)) as usize
}

/// Pulses the pattern fired into each zenith ring: lines in the ring times
/// pulses per line (`phi_count` if `pulses_per_line` is None).
pub fn expected_per_zenith(pattern: &ScanPattern, zenith_edges: &[f64], pulses_per_line: Option<usize>) -> Vec<f64> {
    let n = pulses_per_line.unwrap_or(pattern.phi_count) as f64;
    histogram(pattern.lines(), zenith_edges).into_iter().map(|c| c as f64 * n).collect()
}

/// Pulses a scan fired into each zenith ring, for streams without the
/// pulses that returned nothing: pulses per line counted on the downward
/// lines in `ground_zenith` (where every pulse returns), falling back to
/// [`pulses_per_line`] if fewer than 10 lines fall there. Each line is
/// shared among the rings it overlaps. `zenith` in the scanner frame.
pub fn fired_pulses_per_ring(zenith: &[f64], pattern: &ScanPattern, zenith_edges: &[f64], shot_stride: usize, ground_zenith: (f64, f64)) -> Vec<f64> {
    let theta = pattern.lines();
    let observed: Vec<f64> = histogram(zenith.iter().copied(), &pattern.line_edges()).into_iter().map(|c| c as f64).collect();
    let ground: Vec<f64> = theta.iter().zip(&observed).filter(|(t, _)| **t >= ground_zenith.0 && **t <= ground_zenith.1).map(|(_, &o)| o).collect();
    let ppl = if ground.len() >= 10 && median(&ground) > 0.0 {
        median(&ground)
    } else {
        pulses_per_line(zenith, pattern, 0.98, shot_stride) as f64
    };
    let half = 0.5 * pattern.theta_delta;
    let nr = zenith_edges.len().saturating_sub(1);
    let mut lines = vec![0.0; nr];
    for &t in &theta {
        let (lo, hi) = (t - half, t + half);
        for (r, line) in lines.iter_mut().enumerate() {
            let overlap = (hi.min(zenith_edges[r + 1]) - lo.max(zenith_edges[r])).max(0.0);
            *line += overlap;
        }
    }
    lines.iter().map(|l| l / (2.0 * half) * ppl).collect()
}

/// Pulses a scan fired into each zenith ring from its returns alone (no
/// scan pattern): returns per degree of zenith on the downward lines in
/// `ground_zenith`, times the part of each ring inside the scan's zenith
/// limits. The lower limit is the `limit_quantile` upper quantile of the
/// returns' zenith (every downward pulse hits the ground); the upper one is
/// `field_of_view` degrees above it, since in open vegetation the most
/// upward pulses return nothing. `None` reads both limits from the returns.
pub fn fired_pulses_from_points(zenith: &[f64], zenith_edges: &[f64], ground_zenith: (f64, f64), limit_quantile: f64, field_of_view: Option<f64>) -> Result<Vec<f64>> {
    let mut z = zenith.to_vec();
    z.sort_by(|a, b| a.total_cmp(b));
    let lo = crate::numeric::quantile_sorted(&z, limit_quantile);
    let hi = crate::numeric::quantile_sorted(&z, 1.0 - limit_quantile);
    let lo = field_of_view.map_or(lo, |fov| hi - fov);
    let (g0, g1) = ground_zenith;
    if g0 < lo || g1 > hi {
        return Err(Error::invalid(format!(
            "ground_zenith ({g0:?}, {g1:?}) is outside the scan's zenith limits ({lo:.1}, {hi:.1}) degrees"
        )));
    }
    let per_degree = zenith.iter().filter(|&&v| v >= g0 && v < g1).count() as f64 / (g1 - g0);
    Ok(zenith_edges.windows(2).map(|e| per_degree * (e[1].min(hi) - e[0].max(lo)).max(0.0)).collect())
}

/// Gap fraction by zenith ring for a stream without its misses: a fired
/// pulse (from the pattern) with no echo above `min_height` is a gap. NaN
/// where the pattern fired nothing.
pub fn gap_fraction_pattern(shots: &Shots, echo_heights: &[f64], pattern: &ScanPattern, min_height: f64, zenith_edges: &[f64], pulses_per_line: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if echo_heights.len() != shots.echo_range.len() {
        return Err(Error::invalid("echo_heights must have one value per echo"));
    }
    let expected = expected_per_zenith(pattern, zenith_edges, Some(pulses_per_line));
    let zen = zenith_deg(&shots.direction);
    let hit_zen = (0..shots.n_shots()).filter(|&s| {
        let (e0, n) = (shots.echo_start[s], shots.echo_count[s] as usize);
        echo_heights[e0..e0 + n].iter().any(|&h| h > min_height)
    });
    let hits = histogram(hit_zen.map(|s| zen[s]), zenith_edges);
    let gap = expected.iter().zip(&hits).map(|(&e, &h)| if e == 0.0 { f64::NAN } else { 1.0 - (h as f64 / e).min(1.0) }).collect();
    let centres = zenith_edges.windows(2).map(|e| 0.5 * (e[0] + e[1])).collect();
    Ok((centres, gap))
}

// ---------------------------------------------------------------- ground plane

/// Ground plane `z = a x + b y + c` through the lowest point of every
/// `cell` grid cell (optionally within `radius` of `centre`), by
/// Huber-weighted least squares over `iterations` reweightings.
pub fn fit_ground_plane(points: &[Point], cell: f64, centre: Option<(f64, f64)>, radius: Option<f64>, iterations: usize) -> Result<[f64; 3]> {
    let pts: Vec<Point> = match (centre, radius) {
        (Some((cx, cy)), Some(r)) => points.iter().filter(|p| (p[0] - cx).hypot(p[1] - cy) <= r).copied().collect(),
        _ => points.to_vec(),
    };
    if pts.len() < 3 {
        return Err(Error::invalid("too few points for a ground plane"));
    }
    // Lowest point per cell, cells in (x key, y key) order as np.lexsort gives.
    let key = |p: &Point| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|&a, &b| key(&pts[a]).cmp(&key(&pts[b])).then(pts[a][2].total_cmp(&pts[b][2])));
    let mut low: Vec<Point> = Vec::new();
    let mut last = None;
    for &i in &order {
        let k = key(&pts[i]);
        if last != Some(k) {
            low.push(pts[i]);
            last = Some(k);
        }
    }
    let mut w = vec![1.0; low.len()];
    let mut coef = [0.0; 3];
    for _ in 0..iterations {
        coef = weighted_plane(&low, &w)?;
        let r: Vec<f64> = low.iter().map(|p| p[2] - (coef[0] * p[0] + coef[1] * p[1] + coef[2])).collect();
        let m = median(&r);
        let mad: Vec<f64> = r.iter().map(|v| (v - m).abs()).collect();
        let s = 1.4826 * median(&mad) + 1e-6;
        for (wi, ri) in w.iter_mut().zip(&r) {
            let u = ri.abs() / (1.345 * s);
            *wi = if u <= 1.0 { 1.0 } else { (1.0 / u).sqrt() };
        }
    }
    Ok(coef)
}

/// Least squares of `w z` on `w [x, y, 1]` (NumPy's `lstsq` on the scaled
/// system), by SVD for the same conditioning.
fn weighted_plane(low: &[Point], w: &[f64]) -> Result<[f64; 3]> {
    let a = nalgebra::DMatrix::from_fn(low.len(), 3, |i, j| w[i] * if j == 2 { 1.0 } else { low[i][j] });
    let b = nalgebra::DVector::from_iterator(low.len(), low.iter().zip(w).map(|(p, wi)| wi * p[2]));
    let svd = a.svd(true, true);
    let x = svd.solve(&b, f64::EPSILON * low.len().max(3) as f64).map_err(|e| Error::invalid(e.to_string()))?;
    Ok([x[0], x[1], x[2]])
}

// ------------------------------------------------------------ gap profiles

/// The arrays of a gap profile pooled over scans (row-major): `hits` is
/// `(rings, sectors, heights)`, `shots` is `(rings, sectors)`; per scan,
/// `scan_hits`, `scan_shots` and `scan_low` are `(rings, sectors)`.
#[derive(Debug, Clone, Copy)]
pub struct GapArrays<'a> {
    pub zenith_edges: &'a [f64],
    pub n_azimuth: usize,
    pub height_bin: f64,
    pub n_heights: usize,
    pub hits: &'a [f64],
    pub shots: &'a [f64],
    pub min_height: f64,
}

impl GapArrays<'_> {
    fn n_rings(&self) -> usize {
        self.zenith_edges.len() - 1
    }

    fn check(&self) -> Result<()> {
        let (nr, na, nh) = (self.n_rings(), self.n_azimuth, self.n_heights);
        if self.hits.len() != nr * na * nh || self.shots.len() != nr * na {
            return Err(Error::invalid("gap profile arrays do not match its rings, sectors and heights"));
        }
        Ok(())
    }

    fn first_bin(&self) -> usize {
        (self.min_height / self.height_bin + 1e-9).floor().max(0.0) as usize
    }

    fn fired(&self) -> Vec<f64> {
        (0..self.n_rings()).map(|r| self.shots[r * self.n_azimuth..(r + 1) * self.n_azimuth].iter().sum()).collect()
    }

    /// Ring centres (degrees).
    pub fn zenith(&self) -> Vec<f64> {
        self.zenith_edges.windows(2).map(|e| 0.5 * (e[0] + e[1])).collect()
    }

    /// Top of each height bin (m).
    pub fn heights(&self) -> Vec<f64> {
        (0..self.n_heights).map(|k| (k + 1) as f64 * self.height_bin).collect()
    }

    /// Gap probability `(rings, heights)`: 1 minus the returns from
    /// `min_height` to the top of each bin over the pulses fired, clipped to
    /// 0-1; NaN rows for rings without pulses.
    pub fn pgap(&self) -> Result<Vec<f64>> {
        self.check()?;
        let (nr, na, nh) = (self.n_rings(), self.n_azimuth, self.n_heights);
        let first = self.first_bin();
        let fired = self.fired();
        let mut p = vec![0.0; nr * nh];
        for r in 0..nr {
            let mut cum = 0.0;
            for k in 0..nh {
                if k >= first {
                    cum += (0..na).map(|a| self.hits[(r * na + a) * nh + k]).sum::<f64>();
                }
                p[r * nh + k] = if fired[r] <= 0.0 { f64::NAN } else { (1.0 - cum / fired[r]).clamp(0.0, 1.0) };
            }
        }
        Ok(p)
    }

    /// One pulse's worth of gap per ring, the floor under a closed ring.
    fn floor(&self) -> Vec<f64> {
        self.fired().iter().map(|f| 1.0 / f.max(1.0)).collect()
    }

    fn hinge_ring(&self) -> Option<usize> {
        searchsorted_right(self.zenith_edges, 57.5).checked_sub(1).filter(|&r| r < self.n_rings())
    }

    /// `-ln` of the floored gap probability, and which rings are complete.
    fn log_gap(&self, p: &[f64]) -> (Vec<f64>, Vec<bool>) {
        let (nr, nh) = (self.n_rings(), self.n_heights);
        let floor = self.floor();
        let lp = (0..nr * nh).map(|i| {
            let v = p[i];
            if v.is_nan() { f64::NAN } else { -v.max(floor[i / nh]).ln() }
        }).collect();
        let ok = (0..nr).map(|r| p[r * nh..(r + 1) * nh].iter().all(|v| v.is_finite())).collect();
        (lp, ok)
    }

    /// Cumulative plant area index below each height: `"hinge"` (the ring
    /// holding 57.5 degrees, Jupp et al. 2009), `"linear"` (least squares of
    /// -ln P on tan of the zenith) or `"weighted"` (Miller's integral).
    pub fn pai_profile(&self, method: &str) -> Result<Vec<f64>> {
        let p = self.pgap()?;
        let (lp, ok) = self.log_gap(&p);
        let nh = self.n_heights;
        let th: Vec<f64> = self.zenith().iter().map(|z| z.to_radians()).collect();
        match method {
            "hinge" => {
                let ring = self.hinge_ring().filter(|&r| ok[r]).ok_or_else(|| Error::invalid("no pulses in the ring holding 57.5 deg"))?;
                Ok(lp[ring * nh..(ring + 1) * nh].iter().map(|v| 1.1 * v).collect())
            }
            "weighted" => {
                let w: Vec<f64> = th.iter().zip(&ok).map(|(t, &o)| if o { t.sin() } else { 0.0 }).collect();
                let wsum: f64 = w.iter().sum();
                Ok((0..nh)
                    .map(|k| {
                        let s: f64 = (0..th.len()).map(|r| {
                            let v = lp[r * nh + k];
                            2.0 * th[r].cos() * (if v.is_nan() { 0.0 } else { v }) * w[r]
                        }).sum();
                        s / wsum
                    })
                    .collect())
            }
            "linear" => Ok(self.linear(&lp, &th, &ok).0),
            _ => Err(Error::invalid("method must be 'hinge', 'linear' or 'weighted'")),
        }
    }

    /// Jupp et al. (2009) linear model: -ln P = PAI_h + (2/pi) PAI_v tan(theta),
    /// per height. Returns the PAI and the mean leaf angle (degrees).
    fn linear(&self, lp: &[f64], th: &[f64], ok: &[bool]) -> (Vec<f64>, Vec<f64>) {
        let nh = self.n_heights;
        let rings: Vec<usize> = (0..th.len()).filter(|&r| ok[r]).collect();
        let x: Vec<f64> = rings.iter().map(|&r| th[r].tan()).collect();
        let a = nalgebra::DMatrix::from_fn(x.len(), 2, |i, j| if j == 0 { 1.0 } else { x[i] });
        let svd = a.svd(true, true);
        let mut pai = Vec::with_capacity(nh);
        let mut mla = Vec::with_capacity(nh);
        for k in 0..nh {
            let y = nalgebra::DVector::from_iterator(rings.len(), rings.iter().map(|&r| lp[r * nh + k]));
            let coef = svd.solve(&y, f64::EPSILON * rings.len().max(2) as f64).unwrap_or_else(|_| nalgebra::DVector::from_element(2, f64::NAN));
            let (ph, pv) = (coef[0], coef[1] * std::f64::consts::PI / 2.0);
            pai.push((ph + pv).max(0.0));
            let (h, v) = (ph.max(0.0), pv.max(0.0));
            mla.push(if h + v > 0.0 { v.atan2(h).to_degrees() } else { f64::NAN });
        }
        (pai, mla)
    }

    /// Plant area volume density: the height derivative of [`Self::pai_profile`].
    pub fn pavd_profile(&self, method: &str) -> Result<Vec<f64>> {
        Ok(gradient(&self.pai_profile(method)?, self.height_bin))
    }
}

/// Clumping index at `zenith` (Lang & Xiang 1986) over scan-sector segments:
/// `ln(mean P) / mean(ln P)`, weighted by pulses, each segment's gap floored
/// at one pulse. `scan_low` (returns below the counting height) may be empty.
pub fn clumping(zenith_edges: &[f64], n_azimuth: usize, scan_hits: &[&[f64]], scan_shots: &[&[f64]], scan_low: &[&[f64]], zenith: f64) -> f64 {
    let Some(ring) = searchsorted_right(zenith_edges, zenith).checked_sub(1) else { return f64::NAN };
    let use_low = scan_low.len() == scan_hits.len();
    let (mut p, mut n) = (Vec::new(), Vec::new());
    for (s, (hits, shots)) in scan_hits.iter().zip(scan_shots).enumerate() {
        for a in 0..n_azimuth {
            let i = ring * n_azimuth + a;
            if i >= shots.len() || shots[i] <= 0.0 {
                continue;
            }
            let h = hits[i] - if use_low { scan_low[s][i] } else { 0.0 };
            let v = (1.0 - h / shots[i]).clamp(0.0, 1.0);
            p.push(v.max(1.0 / shots[i]));
            n.push(shots[i]);
        }
    }
    if p.is_empty() {
        return f64::NAN;
    }
    let total: f64 = n.iter().sum();
    let mean_p = p.iter().zip(&n).map(|(a, b)| a * b).sum::<f64>() / total;
    let mean_lnp = p.iter().zip(&n).map(|(a, b)| a.ln() * b).sum::<f64>() / total;
    if mean_lnp < 0.0 { mean_p.ln() / mean_lnp } else { f64::NAN }
}

/// A gap profile's plot summary (see the Python `GapProfile.report`).
#[derive(Debug, Clone)]
pub struct GapReport {
    pub saturated: bool,
    pub gap_57: f64,
    pub pai_hinge: f64,
    pub pai_linear: f64,
    pub pai_weighted: f64,
    pub mla_linear: f64,
    pub clumping: f64,
    pub pai_hinge_corrected: f64,
    pub canopy_height: f64,
    pub closure_57: f64,
    pub cover: f64,
    pub cover_zenith: f64,
    pub n_scans: usize,
    pub pulses: f64,
    pub height: Vec<f64>,
    pub pai_hinge_profile: Vec<f64>,
    pub pavd_hinge: Vec<f64>,
    pub pai_linear_profile: Vec<f64>,
    pub pavd_linear: Vec<f64>,
}

/// Summarise a gap profile: effective PAI three ways, mean leaf angle,
/// clumping and its corrected PAI, canopy height (where the hinge PAI
/// reaches `top_fraction` of its total), closure at 57.5 degrees, cover at
/// the steepest complete ring, and whether the hinge ring is saturated
/// (less than `saturation_gap` of its pulses got through).
pub fn gap_report(g: &GapArrays, scan_hits: &[&[f64]], scan_shots: &[&[f64]], scan_low: &[&[f64]], top_fraction: f64, saturation_gap: f64) -> Result<GapReport> {
    let p = g.pgap()?;
    let nh = g.n_heights;
    let th: Vec<f64> = g.zenith().iter().map(|z| z.to_radians()).collect();
    let (lp, ok) = g.log_gap(&p);
    let hinge = g.pai_profile("hinge")?;
    let (linear, mla) = g.linear(&lp, &th, &ok);
    let weighted = g.pai_profile("weighted")?;
    let omega = clumping(g.zenith_edges, g.n_azimuth, scan_hits, scan_shots, scan_low, 57.5);
    let total = *hinge.last().unwrap_or(&0.0);
    let heights = g.heights();
    let top = if total > 0.0 { heights[searchsorted_left(&hinge, top_fraction * total).min(nh - 1)] } else { 0.0 };
    let ring = g.hinge_ring().unwrap_or(0);
    let steep = ok.iter().position(|&o| o).unwrap_or(0);
    let gap57 = p[ring * nh + nh - 1];
    Ok(GapReport {
        saturated: gap57 < saturation_gap,
        gap_57: gap57,
        pai_hinge: total,
        pai_linear: *linear.last().unwrap_or(&f64::NAN),
        pai_weighted: *weighted.last().unwrap_or(&f64::NAN),
        mla_linear: *mla.last().unwrap_or(&f64::NAN),
        clumping: omega,
        pai_hinge_corrected: if omega != 0.0 && omega.is_finite() { total / omega } else { f64::NAN },
        canopy_height: top,
        closure_57: 1.0 - gap57,
        cover: 1.0 - p[steep * nh + nh - 1],
        cover_zenith: g.zenith()[steep],
        n_scans: scan_hits.len(),
        pulses: g.shots.iter().sum(),
        pavd_hinge: gradient(&hinge, g.height_bin),
        pavd_linear: gradient(&linear, g.height_bin),
        height: heights,
        pai_hinge_profile: hinge,
        pai_linear_profile: linear,
    })
}

// ------------------------------------------------------------ height profiles

/// `np.histogram` of heights on `0, bin_size, ...` up to `max_height` (the
/// highest finite height if None). Returns the bin bottoms and counts.
pub fn vertical_profile(heights: &[f64], bin_size: f64, max_height: Option<f64>) -> (Vec<f64>, Vec<u64>) {
    let top = max_height.unwrap_or_else(|| heights.iter().copied().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max));
    let edges = arange(0.0, top + bin_size, bin_size);
    let counts = histogram(heights.iter().copied(), &edges);
    (edges[..edges.len().saturating_sub(1)].to_vec(), counts)
}

/// A density grid's voxel arrays, `[k, j, i]` (z, y, x) row-major.
#[derive(Debug, Clone, Copy)]
pub struct GridArrays<'a> {
    pub shape: [usize; 3],
    pub origin: [f64; 3],
    pub voxel_size: f64,
    pub n_rays: &'a [f64],
    pub n_hits: &'a [f64],
    pub path_length: &'a [f64],
    pub density: &'a [f64],
}

impl GridArrays<'_> {
    /// Height of each voxel centre above the terrain.
    pub fn height_above(&self, dtm: &Raster) -> Vec<f64> {
        let [nz, ny, nx] = self.shape;
        let v = self.voxel_size;
        let mut out = Vec::with_capacity(nz * ny * nx);
        for k in 0..nz {
            let z = self.origin[2] + (k as f64 + 0.5) * v;
            for j in 0..ny {
                let y = self.origin[1] + (j as f64 + 0.5) * v;
                for i in 0..nx {
                    let x = self.origin[0] + (i as f64 + 0.5) * v;
                    out.push(z - dtm.sample(x, y));
                }
            }
        }
        out
    }

    /// `density` with voxels whose centre is less than `margin` above the
    /// terrain set to NaN, and the mean of the rest per layer.
    pub fn mask_ground(&self, dtm: &Raster, margin: f64) -> (Vec<f64>, Vec<f64>) {
        let h = self.height_above(dtm);
        let density: Vec<f64> = self.density.iter().zip(&h).map(|(&d, &hv)| if hv < margin { f64::NAN } else { d }).collect();
        let layer = self.shape[1] * self.shape[2];
        let profile = density.chunks(layer.max(1)).map(|row| nanmean(row.iter().copied())).collect();
        (density, profile)
    }

    /// Plant area density by height above the terrain: pooled (`2 sum hits /
    /// sum path` per bin) or the mean of voxel densities.
    pub fn profile_above_ground(&self, dtm: &Raster, bin_size: f64, max_height: Option<f64>, margin: f64, pooled: bool) -> (Vec<f64>, Vec<f64>) {
        let h = self.height_above(dtm);
        let ok: Vec<bool> = (0..h.len())
            .map(|i| h[i] >= margin && self.n_rays[i] > 0.0 && (pooled || self.density[i].is_finite()))
            .collect();
        let top = max_height.unwrap_or_else(|| {
            let m = h.iter().zip(&ok).filter(|(_, &o)| o).map(|(&v, _)| v).fold(f64::NEG_INFINITY, f64::max);
            if m.is_finite() { m } else { bin_size }
        });
        let edges = arange(0.0, top + bin_size, bin_size);
        let nb = edges.len().saturating_sub(1);
        let (mut a, mut b) = (vec![0.0; nb], vec![0.0; nb]);
        for i in 0..h.len() {
            if !ok[i] {
                continue;
            }
            // np.digitize(h, edges) - 1: edges[k] <= h < edges[k + 1].
            let k = searchsorted_right(&edges, h[i]) as i64 - 1;
            if k < 0 || k as usize >= nb {
                continue;
            }
            let k = k as usize;
            if pooled {
                a[k] += self.n_hits[i];
                b[k] += self.path_length[i];
            } else {
                a[k] += self.density[i];
                b[k] += 1.0;
            }
        }
        let pad = a.iter().zip(&b).map(|(&x, &y)| {
            if pooled {
                if y > 0.0 { 2.0 * x / y } else { f64::NAN }
            } else {
                x / y
            }
        }).collect();
        (edges[..nb].to_vec(), pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_plane_recovers_a_tilted_plane_under_clutter() {
        let mut pts = Vec::new();
        for i in 0..60 {
            for j in 0..60 {
                let (x, y) = (i as f64 * 0.5 - 15.0, j as f64 * 0.5 - 15.0);
                pts.push([x, y, 0.1 * x - 0.05 * y + 2.0]);
                if (i * 7 + j * 3) % 11 == 0 {
                    pts.push([x, y, 0.1 * x - 0.05 * y + 7.0]);
                }
            }
        }
        let c = fit_ground_plane(&pts, 1.0, None, None, 20).unwrap();
        assert!((c[0] - 0.1).abs() < 1e-6 && (c[1] + 0.05).abs() < 1e-6 && (c[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn fired_from_points_reads_the_downward_rate() {
        // 10 pulses per degree between 30 and 130 degrees, all returned.
        let zen: Vec<f64> = (0..1000).map(|i| 30.05 + i as f64 * 0.1).collect();
        let f = fired_pulses_from_points(&zen, &[25.0, 30.0, 40.0, 60.0], (100.0, 125.0), 1e-5, None).unwrap();
        assert!(f[0].abs() < 1.0 && (f[1] - 100.0).abs() < 1.0 && (f[2] - 200.0).abs() < 1e-9);
    }

    #[test]
    fn a_closed_ring_is_floored_at_one_pulse() {
        let edges = [50.0, 55.0, 60.0];
        let hits = [10.0, 0.0, 10.0, 0.0]; // 2 rings, 1 sector, 2 heights
        let shots = [10.0, 10.0];
        let g = GapArrays { zenith_edges: &edges, n_azimuth: 1, height_bin: 1.0, n_heights: 2, hits: &hits, shots: &shots, min_height: 0.0 };
        let pai = g.pai_profile("hinge").unwrap();
        assert!((pai[1] - 1.1 * (10.0f64).ln()).abs() < 1e-12);
    }
}
