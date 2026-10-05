// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Area-based metrics and plant area index of two surveys compared.
//!
//! **Metrics.** The metrics of [`crate::als::metrics`] are computed per grid
//! cell for both surveys in one pass (the second moved into the first's
//! frame, optionally thinned to a common pulse density), from the same
//! canonical order of each cell's returns. Whether a metric changed is
//! tested as the canopy heights of [`super::surface`] are: if nothing
//! changed, the pulses of both surveys that fell in one small square of the
//! cell (a stratum of side `stratum`) sample the same canopy, and any
//! reassignment of them between the surveys, keeping each survey's number of
//! pulses in each square, is as likely as the observed one. The
//! distribution of `metric(b) - metric(a)` over `permutations` such
//! reassignments (a stratified permutation test; Pitman 1937) is the change
//! sampling alone would give; its mean is the sampling bias and its central
//! `confidence` interval bounds the change expected without any. Keeping the
//! pulses in their squares keeps the pattern in which each survey sampled
//! the cell (flight lines, overlaps), which a resampling of the cell's pulses
//! as independent draws would ignore. Returns of one pulse move together.
//! Differences between the sensors (footprint, sensitivity) that make the
//! surveys see the same canopy differently are not sampling and are not
//! covered: they bias metrics systematically, and harmonisation removes
//! only the part due to pulse density.
//!
//! **Plant area index.** Two [`ProfileGrid`]s on one grid are compared cell
//! by cell. With `W` pulses in a cell, a share `P` of them reaching the
//! ground and a mean extinction `k`, `PAI = -ln P / k`; the binomial
//! variance of `P` gives `var(ln P) = (1 - P) / (W P)` by the delta method,
//! and so the standard deviation `sqrt((1 - P) / (W P)) / k`. A cell no pulse
//! crossed is saturated: its PAI is a lower bound, and its change is not
//! classed as gain or loss. The profile of an area is compared layer by
//! layer from its pooled counts in the same way.

use std::collections::BTreeMap;

use crate::als::{catalog_grid, plan, run, workers_for_estimates, Catalog, Layout, BYTES_PER_POINT};
use crate::als::canopy::{column_pad, column_pai, ProfileGrid};
use crate::als::metrics::{assemble, chunk_cells, compute, metric_names, metrics_buffer, select, Available, HeightSource, MetricParams, Return};
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::raster::Raster;
use crate::PointCloud;

use super::surface::{chunk_dtm, ground_of, shift_crop, DtmKind, PLANE_RADIUS};
use super::{est_other, harmonise_pair, mix, pulse_keys, unit, z_of, Alignment, Harmonise};

const HEIGHT: &str = "__sylva_height";

/// Settings of [`metric_change`].
#[derive(Debug, Clone)]
pub struct MetricChangeParams {
    pub metrics: MetricParams,
    /// Metrics to compare (all when None).
    pub names: Option<Vec<String>>,
    pub resolution: f64,
    pub dtm_resolution: f64,
    pub dtm: DtmKind,
    /// Permutations per cell (0 for none: no level of detection).
    pub permutations: usize,
    /// Side (m) of the squares within which pulses are permuted.
    pub stratum: f64,
    pub seed: u64,
    pub confidence: f64,
    pub harmonise: Option<Harmonise>,
}

impl Default for MetricChangeParams {
    fn default() -> Self {
        MetricChangeParams { metrics: MetricParams::default(), names: None, resolution: 20.0, dtm_resolution: 1.0, dtm: DtmKind::Plane, permutations: 100, stratum: 2.0, seed: 0, confidence: 0.95, harmonise: None }
    }
}

/// Metric rasters of two surveys and their differences.
#[derive(Debug, Clone)]
pub struct MetricChange {
    pub names: Vec<String>,
    pub a: Vec<Raster>,
    pub b: Vec<Raster>,
    pub difference: Vec<Raster>,
    /// Mean of the permutation distribution, its interval, half-width and
    /// standard deviation.
    pub bias: Vec<Raster>,
    pub lower: Vec<Raster>,
    pub upper: Vec<Raster>,
    pub lod: Vec<Raster>,
    pub sigma: Vec<Raster>,
    /// Codes of [`super::surface::CLASSES`], per metric.
    pub classes: Vec<Vec<u8>>,
}

/// A deterministic stream of uniform numbers.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = mix(self.0);
        unit(self.0)
    }
}

/// One survey's returns in a cell: the returns, their pulse keys and positions.
#[derive(Debug, Clone, Default)]
pub struct CellReturns {
    pub returns: Vec<Return>,
    pub keys: Vec<u64>,
    pub xy: Vec<[f64; 2]>,
}

impl CellReturns {
    /// Pulses as `(stratum, returns)`, the stratum from the pulse's first
    /// return in canonical order, in order of their keys.
    fn pulses(&self, stratum: f64) -> Vec<((i64, i64), Vec<Return>)> {
        let mut by: BTreeMap<u64, ((i64, i64), Vec<Return>)> = BTreeMap::new();
        for (i, r) in self.returns.iter().enumerate() {
            let s = ((self.xy[i][0] / stratum).floor() as i64, (self.xy[i][1] / stratum).floor() as i64);
            by.entry(self.keys[i]).or_insert_with(|| (s, Vec::new())).1.push(*r);
        }
        by.into_values().collect()
    }
}

/// The change of each metric `pick` a cell would show from sampling alone.
#[derive(Debug, Clone, PartialEq)]
struct CellNull {
    a: Vec<f64>,
    b: Vec<f64>,
    mean: Vec<f64>,
    sd: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// The metrics of both surveys in a cell and the stratified permutation
/// distribution of their differences; see the module documentation.
#[allow(clippy::too_many_arguments)]
fn cell_null(a: &CellReturns, b: &CellReturns, avail: Available, params: &MetricParams, pick: &[usize], perms: usize, stratum: f64, alpha: f64, seed: u64) -> CellNull {
    let metrics = |r: &[Return]| -> Vec<f64> {
        let mut v = r.to_vec();
        let full = compute(&mut v, avail, params);
        pick.iter().map(|&j| full[j]).collect()
    };
    let (va, vb) = (metrics(&a.returns), metrics(&b.returns));
    let k = pick.len();
    let nan = vec![f64::NAN; k];
    if perms == 0 || a.returns.is_empty() || b.returns.is_empty() {
        return CellNull { a: va, b: vb, mean: nan.clone(), sd: nan.clone(), lower: nan.clone(), upper: nan };
    }
    // Pulses of both surveys by stratum; each stratum keeps its count per survey.
    let mut strata: BTreeMap<(i64, i64), (usize, Vec<Vec<Return>>)> = BTreeMap::new();
    for (s, r) in a.pulses(stratum) {
        let e = strata.entry(s).or_default();
        e.0 += 1;
        e.1.push(r);
    }
    for (s, r) in b.pulses(stratum) {
        strata.entry(s).or_default().1.push(r);
    }
    let mut groups: Vec<(usize, Vec<Vec<Return>>)> = strata.into_values().collect();
    let mut st = Stream(seed);
    let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(perms); k];
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for _ in 0..perms {
        ra.clear();
        rb.clear();
        for (na, pulses) in groups.iter_mut() {
            // Fisher-Yates on the stratum's pulses; the first na go to a.
            for i in (1..pulses.len()).rev() {
                let j = ((st.next() * (i + 1) as f64) as usize).min(i);
                pulses.swap(i, j);
            }
            for (i, p) in pulses.iter().enumerate() {
                if i < *na { ra.extend_from_slice(p) } else { rb.extend_from_slice(p) }
            }
        }
        let (ma, mb) = (metrics(&ra), metrics(&rb));
        for q in 0..k {
            let d = mb[q] - ma[q];
            if d.is_finite() {
                draws[q].push(d);
            }
        }
    }
    let mut out = CellNull { a: va, b: vb, mean: nan.clone(), sd: nan.clone(), lower: nan.clone(), upper: nan };
    for (q, d) in draws.iter_mut().enumerate() {
        if 2 * d.len() < perms || d.len() < 2 {
            continue;
        }
        d.sort_by(|x, y| x.total_cmp(y));
        let n = d.len() as f64;
        let mean = d.iter().sum::<f64>() / n;
        out.mean[q] = mean;
        out.sd[q] = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        out.lower[q] = crate::util::numeric::quantile_sorted(d, alpha / 2.0);
        out.upper[q] = crate::util::numeric::quantile_sorted(d, 1.0 - alpha / 2.0);
    }
    out
}

/// A cell's index on the grid, whether its centre is in the chunk's core,
/// and its returns.
type Cell = (usize, bool, CellReturns);

/// The returns of a chunk's cloud per grid cell, with their pulse keys.
fn per_cell(cat: &Catalog, chunk: &crate::als::Chunk, cloud: &PointCloud, grid: &Raster, params: &MetricParams) -> Result<Vec<Cell>> {
    let Some(cells) = chunk_cells(cat, chunk, cloud, grid, &HeightSource::Attribute(HEIGHT.into()), params)? else { return Ok(Vec::new()) };
    let keys = pulse_keys(cloud);
    let get = |name: &str, i: usize| cloud.attr(name).map_or(f64::NAN, |a| a.get_f64(i));
    Ok((0..cells.cells.len())
        .map(|g| {
            let (s, e) = (cells.starts[g], cells.starts[g + 1]);
            let mut c = CellReturns::default();
            for k in s..e {
                let i = cells.order[k];
                c.returns.push(Return { z: cells.heights[k], intensity: get("intensity", i), return_number: get("return_number", i), classification: get("classification", i) });
                c.keys.push(keys[i]);
                c.xy.push([cloud.xyz[i][0], cloud.xyz[i][1]]);
            }
            (cells.cells[g], cells.central[g], c)
        })
        .collect())
}

/// Heights above the survey's own DTM as an attribute; None without ground.
fn with_heights(mut cloud: PointCloud, grid: &Raster, outer: &[f64; 4], kind: DtmKind, reach: f64) -> Result<Option<PointCloud>> {
    let ground = ground_of(&cloud);
    let Some((d, _)) = chunk_dtm(&ground, grid, outer, kind, 0.0, 0.0, reach)? else { return Ok(None) };
    let h = super::surface::heights(&cloud, &d, (grid.xmin, grid.ymin));
    cloud.attrs.insert(HEIGHT.into(), Attr::F64(h));
    Ok(Some(cloud))
}

/// Compare area-based metrics of two surveys; see the module documentation.
pub fn metric_change(cat_a: &Catalog, cat_b: &Catalog, alignment: Option<&Alignment>, p: &MetricChangeParams, layout: Layout, buffer: f64, workers: usize) -> Result<MetricChange> {
    p.metrics.check()?;
    cat_b.check_usable()?;
    if let Some(h) = &p.harmonise {
        h.check()?;
    }
    if !(p.stratum.is_finite() && p.stratum > 0.0) {
        return Err(Error::invalid(format!("stratum must be a positive number of metres, got {}", p.stratum)));
    }
    z_of(p.confidence)?;
    let alpha = 1.0 - p.confidence;
    let grid = catalog_grid(cat_a, p.resolution)?;
    let dgrid = catalog_grid(cat_a, p.dtm_resolution)?;
    let avail = Available::ALL;
    let all = metric_names(avail, &p.metrics);
    let pick = select(&all, p.names.as_deref())?;
    let k = pick.len();
    // Whole cells, and the ground planes of the DTM cells under them.
    let mut buffer = metrics_buffer(buffer, p.resolution).max(0.5 * p.resolution + 2.0 * p.dtm_resolution + PLANE_RADIUS);
    if let Some(h) = &p.harmonise {
        buffer = buffer.max(h.cell + 2.0 * p.resolution);
    }
    let grow = alignment.map_or(0.0, |a| a.max_horizontal() + 0.5);
    let chunks = plan(cat_a, layout, buffer)?;
    let est: Vec<u64> = chunks.iter().map(|c| c.est_points + est_other(cat_b, c, grow)).collect();
    let w = workers_for_estimates(&est, workers, BYTES_PER_POINT)?;
    let parts = run(cat_a, &chunks, w, "metric change", |chunk, data| {
        let (braw, _) = super::read_other(cat_b, chunk, grow, None)?;
        let bbuf = vec![true; braw.len()];
        let (a, b) = match &p.harmonise {
            Some(h) => harmonise_pair((data.cloud, data.buffer), (braw, bbuf), h),
            None => ((data.cloud, data.buffer), (braw, bbuf)),
        };
        let b = shift_crop(b.0, chunk, alignment);
        let reach = PLANE_RADIUS;
        let (Some(ca), Some(cb)) = (with_heights(a.0, &dgrid, &chunk.outer, p.dtm, reach)?, with_heights(b.0, &dgrid, &chunk.outer, p.dtm, reach)?) else { return Ok(None) };
        let (xa, xb) = (per_cell(cat_a, chunk, &ca, &grid, &p.metrics)?, per_cell(cat_a, chunk, &cb, &grid, &p.metrics)?);
        // Merge the two sorted cell lists.
        let mut cells = Vec::new();
        let mut central = Vec::new();
        let mut values = Vec::new();
        let (mut i, mut j) = (0, 0);
        let empty = CellReturns::default();
        while i < xa.len() || j < xb.len() {
            let ci = xa.get(i).map_or(usize::MAX, |c| c.0);
            let cj = xb.get(j).map_or(usize::MAX, |c| c.0);
            let cell = ci.min(cj);
            let (ra, cen) = if ci == cell {
                i += 1;
                (&xa[i - 1].2, xa[i - 1].1)
            } else {
                (&empty, false)
            };
            let (rb, cen_b) = if cj == cell {
                j += 1;
                (&xb[j - 1].2, xb[j - 1].1)
            } else {
                (&empty, false)
            };
            let nl = cell_null(ra, rb, avail, &p.metrics, &pick, p.permutations, p.stratum, alpha, mix(p.seed ^ mix(cell as u64)));
            cells.push(cell);
            central.push(cen || cen_b);
            for q in 0..k {
                values.extend([nl.a[q], nl.b[q], nl.mean[q], nl.sd[q], nl.lower[q], nl.upper[q]]);
            }
        }
        Ok(Some((cells, central, values)))
    })?;
    let parts: Vec<_> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() {
        return Err(Error::invalid("no chunk has 3 ground returns (classification 2) in both surveys; classify ground first (als.classify_ground)"));
    }
    let layers = assemble(&grid, 6 * k, &parts);
    let n = grid.nrows * grid.ncols;
    let mut out = MetricChange { names: pick.iter().map(|&j| all[j].clone()).collect(), a: Vec::new(), b: Vec::new(), difference: Vec::new(), bias: Vec::new(), lower: Vec::new(), upper: Vec::new(), lod: Vec::new(), sigma: Vec::new(), classes: Vec::new() };
    for q in 0..k {
        let l = &layers[6 * q..6 * q + 6];
        let mut d = vec![f64::NAN; n];
        let mut lod = vec![f64::NAN; n];
        let mut c = vec![0u8; n];
        for i in 0..n {
            let di = l[1].data[i] - l[0].data[i];
            if !di.is_finite() {
                continue;
            }
            d[i] = di;
            let (lo, hi) = (l[4].data[i], l[5].data[i]);
            if lo.is_finite() && hi.is_finite() {
                lod[i] = 0.5 * (hi - lo);
                c[i] = if di > hi { 2 } else if di < lo { 3 } else { 1 };
            }
        }
        let r = |v: Vec<f64>| Raster { data: v, ..grid.clone() };
        out.a.push(l[0].clone());
        out.b.push(l[1].clone());
        out.difference.push(r(d));
        out.bias.push(l[2].clone());
        out.sigma.push(l[3].clone());
        out.lower.push(l[4].clone());
        out.upper.push(l[5].clone());
        out.lod.push(r(lod));
        out.classes.push(c);
    }
    Ok(out)
}

// ------------------------------------------------------------------ PAI

/// Plant area index of two surveys compared per cell.
#[derive(Debug, Clone, PartialEq)]
pub struct PaiChange {
    pub pai_a: Raster,
    pub pai_b: Raster,
    pub sigma_a: Raster,
    pub sigma_b: Raster,
    pub difference: Raster,
    pub sigma: Raster,
    pub lod: Raster,
    /// Codes of [`PAI_CLASSES`].
    pub classes: Vec<u8>,
}

/// Names of the PAI change classes, by code.
pub const PAI_CLASSES: [&str; 5] = ["no_data", "below_detection", "gain", "loss", "saturated"];

fn same_grid(a: &ProfileGrid, b: &ProfileGrid) -> Result<()> {
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
    if a.nx != b.nx || a.ny != b.ny || a.nz != b.nz || !close(a.xmin, b.xmin) || !close(a.ymin, b.ymin) || !close(a.resolution, b.resolution) || !close(a.min_height, b.min_height) || !close(a.bin_size, b.bin_size) {
        return Err(Error::invalid(format!(
            "the two profiles must share their grid and layers: {} x {} x {} cells of {} m from ({}, {}), layers of {} m from {} m, against {} x {} x {} of {} m from ({}, {}), {} m from {} m; make both with the same resolution, bin_size, min_height and max_height over the same area",
            a.nx, a.ny, a.nz, a.resolution, a.xmin, a.ymin, a.bin_size, a.min_height, b.nx, b.ny, b.nz, b.resolution, b.xmin, b.ymin, b.bin_size, b.min_height
        )));
    }
    Ok(())
}

/// PAI of one column and its standard deviation; whether it is saturated.
fn pai_sd(w: &[f64], wk: &[f64]) -> (f64, f64, bool) {
    let total: f64 = w.iter().sum();
    let tk: f64 = wk.iter().sum();
    if !(total > 0.0 && tk > 0.0) {
        return (f64::NAN, f64::NAN, false);
    }
    let pai = column_pai(w, wk);
    let p = w[0] / total;
    if p <= 0.0 {
        return (pai, f64::NAN, true);
    }
    let k = tk / total;
    (pai, ((1.0 - p) / (total * p)).sqrt() / k, false)
}

fn column(g: &ProfileGrid, v: &[f64], cell: usize) -> Vec<f64> {
    let per = g.nx * g.ny;
    (0..g.nz + 2).map(|l| v[l * per + cell]).collect()
}

/// Compare the PAI of two profile grids; see the module documentation.
pub fn pai_change(a: &ProfileGrid, b: &ProfileGrid, confidence: f64) -> Result<PaiChange> {
    same_grid(a, b)?;
    let z = z_of(confidence)?;
    let per = a.nx * a.ny;
    let grid = Raster::filled(a.ny, a.nx, a.xmin, a.ymin, a.resolution, f64::NAN);
    let mut v = vec![vec![f64::NAN; per]; 7];
    let mut cls = vec![0u8; per];
    for c in 0..per {
        let (pa, sa, sat_a) = pai_sd(&column(a, &a.weight, c), &column(a, &a.weight_k, c));
        let (pb, sb, sat_b) = pai_sd(&column(b, &b.weight, c), &column(b, &b.weight_k, c));
        v[0][c] = pa;
        v[1][c] = pb;
        v[2][c] = sa;
        v[3][c] = sb;
        let d = pb - pa;
        if !d.is_finite() {
            continue;
        }
        v[4][c] = d;
        if sat_a || sat_b {
            cls[c] = 4;
            continue;
        }
        let s = sa.hypot(sb);
        v[5][c] = s;
        v[6][c] = z * s;
        cls[c] = if d > z * s { 2 } else if d < -z * s { 3 } else { 1 };
    }
    let r = |d: Vec<f64>| Raster { data: d, ..grid.clone() };
    let mut it = v.into_iter();
    let mut nx = || r(it.next().expect("layer"));
    Ok(PaiChange { pai_a: nx(), pai_b: nx(), sigma_a: nx(), sigma_b: nx(), difference: nx(), sigma: nx(), lod: nx(), classes: cls })
}

/// The pooled plant area density profile of an area (the cells where
/// `mask` is true, all when None) in both surveys: layer centre heights, PAD
/// of each survey, their difference and its standard deviation (NaN for a
/// layer saturated in either survey).
#[allow(clippy::type_complexity)]
pub fn profile_change(a: &ProfileGrid, b: &ProfileGrid, mask: Option<&[bool]>) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    same_grid(a, b)?;
    let per = a.nx * a.ny;
    if let Some(m) = mask {
        if m.len() != per {
            return Err(Error::invalid(format!("the mask has {} cells, the grid {per}", m.len())));
        }
    }
    let pooled = |g: &ProfileGrid, v: &[f64]| -> Vec<f64> {
        (0..g.nz + 2).map(|l| (0..per).filter(|&c| mask.is_none_or(|m| m[c])).map(|c| v[l * per + c]).sum()).collect()
    };
    let sd = |w: &[f64], wk: &[f64], dz: f64| -> Vec<f64> {
        let nz = w.len() - 2;
        let (mut below, mut below_k) = (w[0], wk[0]);
        (1..=nz)
            .map(|i| {
                let e = below + w[i];
                let ek = below_k + wk[i];
                let t = if e > 0.0 { below / e } else { f64::NAN };
                let out = if t > 0.0 && ek > 0.0 { ((1.0 - t) / (e * t)).sqrt() / (ek / e * dz) } else { f64::NAN };
                below = e;
                below_k = ek;
                out
            })
            .collect()
    };
    let (wa, wka, wb, wkb) = (pooled(a, &a.weight), pooled(a, &a.weight_k), pooled(b, &b.weight), pooled(b, &b.weight_k));
    let (pa, pb) = (column_pad(&wa, &wka, a.bin_size), column_pad(&wb, &wkb, b.bin_size));
    let (sa, sb) = (sd(&wa, &wka, a.bin_size), sd(&wb, &wkb, b.bin_size));
    let heights = (0..a.nz).map(|i| a.min_height + (i as f64 + 0.5) * a.bin_size).collect();
    let diff = pa.iter().zip(&pb).map(|(x, y)| y - x).collect();
    let sigma = sa.iter().zip(&sb).map(|(x, y)| x.hypot(*y)).collect();
    Ok((heights, pa, pb, diff, sigma))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(w0: f64, w1: f64, w2: f64) -> ProfileGrid {
        // One layer; weights: below, in the layer, above.
        ProfileGrid { xmin: 0.0, ymin: 0.0, resolution: 10.0, nx: 1, ny: 1, nz: 1, min_height: 1.0, bin_size: 20.0, weight: vec![w0, w1, w2], weight_k: vec![w0 * 0.5, w1 * 0.5, w2 * 0.5], n_skipped: 0 }
    }

    #[test]
    fn pai_change_has_a_binomial_uncertainty() {
        let (a, b) = (grid(400.0, 600.0, 0.0), grid(200.0, 800.0, 0.0));
        let c = pai_change(&a, &b, 0.95).unwrap();
        let pai = |p: f64| -p.ln() / 0.5;
        assert!((c.pai_a.data[0] - pai(0.4)).abs() < 1e-12 && (c.pai_b.data[0] - pai(0.2)).abs() < 1e-12);
        let sd = |p: f64| ((1.0 - p) / (1000.0 * p)).sqrt() / 0.5;
        assert!((c.sigma.data[0] - sd(0.4).hypot(sd(0.2))).abs() < 1e-12);
        assert_eq!(c.classes[0], 2);
        let sat = pai_change(&a, &grid(0.0, 1000.0, 0.0), 0.95).unwrap();
        assert_eq!(sat.classes[0], 4);
        let mut other = a.clone();
        other.bin_size = 10.0;
        assert!(pai_change(&a, &other, 0.95).is_err());
        let (h, pa, pb, d, s) = profile_change(&a, &b, None).unwrap();
        assert_eq!(h, vec![11.0]);
        assert!((d[0] - (pb[0] - pa[0])).abs() < 1e-12 && s[0] > 0.0);
        // The profile's single layer holds the whole PAI.
        assert!((pa[0] * 20.0 - c.pai_a.data[0]).abs() < 1e-9);
    }

    #[test]
    fn the_permutation_null_is_calibrated_and_detects_a_shift() {
        let params = MetricParams::default();
        let avail = Available { intensity: false, returns: false, classification: false };
        let names = metric_names(avail, &params);
        let pick = select(&names, Some(&["zmean".to_string(), "zq95".to_string()])).unwrap();
        let mut s = Stream(3);
        // Pulses over a 20 m cell whose canopy height rises to the east.
        let make = |n: usize, shift: f64, s: &mut Stream, key0: u64| -> CellReturns {
            let mut c = CellReturns::default();
            for k in 0..n {
                let (x, y) = (20.0 * s.next(), 20.0 * s.next());
                c.returns.push(Return { z: 0.5 * x + 3.0 * s.next() + shift, intensity: f64::NAN, return_number: f64::NAN, classification: f64::NAN });
                c.keys.push(key0 + k as u64);
                c.xy.push([x, y]);
            }
            c
        };
        let (mut outside, trials) = (0, 200);
        for t in 0..trials {
            let (a, b) = (make(150, 0.0, &mut s, 0), make(60, 0.0, &mut s, 1_000_000));
            let nl = cell_null(&a, &b, avail, &params, &pick, 200, 2.0, 0.05, t);
            let d = nl.b[0] - nl.a[0];
            if d < nl.lower[0] || d > nl.upper[0] {
                outside += 1;
            }
        }
        assert!((outside as f64 / trials as f64) < 0.09, "{outside}");
        let (a, b) = (make(150, 0.0, &mut s, 0), make(60, 1.5, &mut s, 1_000_000));
        let nl = cell_null(&a, &b, avail, &params, &pick, 200, 2.0, 0.05, 1);
        assert!(nl.b[0] - nl.a[0] > nl.upper[0], "{nl:?}");
        assert_eq!(nl, cell_null(&a, &b, avail, &params, &pick, 200, 2.0, 0.05, 1));
        let none = cell_null(&a, &CellReturns::default(), avail, &params, &pick, 200, 2.0, 0.05, 1);
        assert!(none.lower.iter().all(|v| v.is_nan()));
    }
}
