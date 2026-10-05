// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One cloud and one plant area density profile from TLS and ALS.
//!
//! **Clouds.** Below a split height above ground the TLS points are kept,
//! above it the ALS returns: the scanner below the canopy samples the stems
//! and the understorey densely, the airborne one the upper crowns it looks
//! down on. The split is a height, or a raster of heights (one per column).
//!
//! **Profiles.** Each instrument's ray-traced voxels are summarised per bin
//! of height above ground over the same area ([`layer_stats`]): the plant
//! area density (PAD) of the bin, the mean number of pulses entering its
//! voxels (unreached voxels count 0, so occlusion lowers it) and the share of
//! its voxels crossed by at least `min_beams` pulses. PAD is the mean of a
//! PAD field over the voxels seen, or pooled, `Σ hits / (G Σ free path)`
//! (a ratio of sums, which weights each voxel by the pulses that reached
//! it: right for a uniform layer, low where shadowed crowns alternate with
//! gaps). [`fuse`] then takes each bin from
//! both, weighted by the pulses (`"beams"`: the number of beams is what the
//! variance of a gap-fraction estimate scales with), by the share of the
//! bin each observed (`"observed"`), or from the better one alone
//! (`"best"`). The weights are reported per bin.

use crate::error::{Error, Result};
use crate::raster::Raster;
use crate::Point;

/// Where the TLS stops and the ALS starts.
#[derive(Debug, Clone)]
pub enum Split {
    Height(f64),
    /// Height per column; NaN cells and points off the grid take `fallback`.
    Raster { raster: Raster, fallback: f64 },
}

impl Split {
    fn at(&self, x: f64, y: f64) -> f64 {
        match self {
            Split::Height(h) => *h,
            Split::Raster { raster, fallback } => {
                let (row, col) = raster.cell_index(x, y);
                if raster.in_bounds(row, col) {
                    let v = raster.get(row as usize, col as usize);
                    if v.is_finite() { v } else { *fallback }
                } else {
                    *fallback
                }
            }
        }
    }
}

/// Which points to keep: those lower than the split with `below`, those at
/// or above it otherwise. Points with a NaN height are dropped.
pub fn select(points: &[Point], heights: &[f64], split: &Split, below: bool) -> Result<Vec<bool>> {
    if points.len() != heights.len() {
        return Err(Error::invalid("one height per point is needed"));
    }
    if let Split::Height(h) = split {
        if !h.is_finite() {
            return Err(Error::invalid("the split height must be finite"));
        }
    }
    Ok(points.iter().zip(heights).map(|(p, &h)| {
        let s = split.at(p[0], p[1]);
        h.is_finite() && s.is_finite() && ((h < s) == below)
    }).collect())
}

/// A voxel grid's fields, arrays in `(nz, ny, nx)` order.
pub struct VoxelFields<'a> {
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nz, ny, nx]`.
    pub shape: [usize; 3],
    pub beams: &'a [f64],
    /// A PAD field (mean estimator), or hits and free path lengths (pooled).
    pub pad: Option<&'a [f64]>,
    pub hits: Option<&'a [f64]>,
    pub path: Option<&'a [f64]>,
}

/// Per bin of height above ground: its bottom, PAD, mean pulses per voxel,
/// share of voxels observed, and the voxels in it.
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub height: Vec<f64>,
    pub pad: Vec<f64>,
    pub beams: Vec<f64>,
    pub observed: Vec<f64>,
    pub n_voxels: Vec<usize>,
}

/// Summarise a voxel grid by height above `dtm` (above the grid floor
/// without one) over the columns of `mask` (`ny * nx`, all if None). With
/// hits and path lengths the PAD is pooled and divided by `g`; otherwise it
/// is the mean of the PAD field over the observed voxels.
pub fn layer_stats(v: &VoxelFields, mask: Option<&[bool]>, dtm: Option<&Raster>, bin_size: f64, min_beams: f64, g: f64) -> Result<LayerStats> {
    let [nz, ny, nx] = v.shape;
    let n = nz * ny * nx;
    if !(bin_size.is_finite() && bin_size > 0.0) || !(v.voxel_size.is_finite() && v.voxel_size > 0.0) {
        return Err(Error::invalid("bin_size and voxel_size must be positive"));
    }
    if !(g.is_finite() && g > 0.0) || !min_beams.is_finite() {
        return Err(Error::invalid("g must be positive and min_beams finite"));
    }
    if v.beams.len() != n || v.pad.is_some_and(|a| a.len() != n) || v.hits.is_some_and(|a| a.len() != n) || v.path.is_some_and(|a| a.len() != n) {
        return Err(Error::invalid("every field must have one value per voxel"));
    }
    let pooled = v.hits.is_some() && v.path.is_some();
    if !pooled && v.pad.is_none() {
        return Err(Error::invalid("give a PAD field, or hits and free path lengths"));
    }
    if mask.is_some_and(|m| m.len() != nx * ny) {
        return Err(Error::invalid(format!("the column mask must have {ny} x {nx} values")));
    }
    let vs = v.voxel_size;
    let ground: Vec<f64> = (0..nx * ny).map(|c| match dtm {
        Some(d) => d.sample(v.origin[0] + ((c % nx) as f64 + 0.5) * vs, v.origin[1] + ((c / nx) as f64 + 0.5) * vs),
        None => v.origin[2],
    }).collect();
    // (voxels, beams, observed, numerator, denominator)
    let mut acc: Vec<(usize, f64, usize, f64, f64)> = Vec::new();
    for k in 0..nz {
        let zc = v.origin[2] + (k as f64 + 0.5) * vs;
        for (c, gz) in ground.iter().enumerate() {
            if mask.is_some_and(|m| !m[c]) || !gz.is_finite() {
                continue;
            }
            let h = zc - gz;
            if h < 0.0 {
                continue;
            }
            let b = (h / bin_size).floor() as usize;
            if b >= acc.len() {
                acc.resize(b + 1, (0, 0.0, 0, 0.0, 0.0));
            }
            let i = k * nx * ny + c;
            let beams = if v.beams[i].is_finite() { v.beams[i] } else { 0.0 };
            let e = &mut acc[b];
            e.0 += 1;
            e.1 += beams;
            if beams >= min_beams && beams > 0.0 {
                if pooled {
                    let (hi, pa) = (v.hits.unwrap()[i], v.path.unwrap()[i]);
                    if hi.is_finite() && pa.is_finite() {
                        e.2 += 1;
                        e.3 += hi;
                        e.4 += pa;
                    }
                } else {
                    let p = v.pad.unwrap()[i];
                    if p.is_finite() {
                        e.2 += 1;
                        e.3 += p;
                        e.4 += 1.0;
                    }
                }
            }
        }
    }
    let mut out = LayerStats::default();
    for (b, e) in acc.iter().enumerate() {
        out.height.push(b as f64 * bin_size);
        out.n_voxels.push(e.0);
        out.beams.push(if e.0 > 0 { e.1 / e.0 as f64 } else { f64::NAN });
        out.observed.push(if e.0 > 0 { e.2 as f64 / e.0 as f64 } else { f64::NAN });
        out.pad.push(if e.4 > 0.0 { e.3 / e.4 / if pooled { g } else { 1.0 } } else { f64::NAN });
    }
    Ok(out)
}

/// How [`fuse`] weights the two instruments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseMode {
    Beams,
    Observed,
    Best,
}

impl FuseMode {
    pub fn parse(s: &str) -> Result<FuseMode> {
        match s {
            "beams" => Ok(FuseMode::Beams),
            "observed" => Ok(FuseMode::Observed),
            "best" => Ok(FuseMode::Best),
            _ => Err(Error::invalid(format!("unknown mode {s:?}; expected 'beams', 'observed' or 'best'"))),
        }
    }
}

/// The fused profile: per bin, PAD, the weight of each instrument (summing
/// to 1; NaN where neither saw the bin) and the inputs.
#[derive(Debug, Clone, Default)]
pub struct Fused {
    pub height: Vec<f64>,
    pub pad: Vec<f64>,
    pub weight_tls: Vec<f64>,
    pub weight_als: Vec<f64>,
    pub tls: LayerStats,
    pub als: LayerStats,
    /// Lowest height above which the ALS carries at least half the weight
    /// in every bin.
    pub split_height: f64,
}

fn padded(s: &LayerStats, n: usize, bin: f64) -> LayerStats {
    let mut o = s.clone();
    while o.height.len() < n {
        o.height.push(o.height.len() as f64 * bin);
        o.pad.push(f64::NAN);
        o.beams.push(f64::NAN);
        o.observed.push(f64::NAN);
        o.n_voxels.push(0);
    }
    o
}

/// Combine two layer summaries on the same bins; `min_observed` is the share
/// of a bin an instrument must have seen for its PAD to count.
pub fn fuse(tls: &LayerStats, als: &LayerStats, bin_size: f64, mode: FuseMode, min_observed: f64) -> Result<Fused> {
    if !(0.0..=1.0).contains(&min_observed) {
        return Err(Error::invalid("min_observed must be in [0, 1]"));
    }
    let n = tls.height.len().max(als.height.len());
    let (t, a) = (padded(tls, n, bin_size), padded(als, n, bin_size));
    let mut f = Fused { height: t.height.clone(), ..Default::default() };
    let usable = |s: &LayerStats, i: usize| s.pad[i].is_finite() && s.observed[i] >= min_observed && s.observed[i] > 0.0;
    for i in 0..n {
        let (ut, ua) = (usable(&t, i), usable(&a, i));
        let score = |s: &LayerStats, ok: bool| if !ok { 0.0 } else if mode == FuseMode::Observed { s.observed[i] } else { s.beams[i] };
        let (st, sa) = (score(&t, ut), score(&a, ua));
        let (wt, wa) = if st + sa <= 0.0 {
            (f64::NAN, f64::NAN)
        } else if mode == FuseMode::Best {
            if st >= sa { (1.0, 0.0) } else { (0.0, 1.0) }
        } else {
            (st / (st + sa), sa / (st + sa))
        };
        let pad = if wt.is_nan() {
            f64::NAN
        } else {
            (if wt > 0.0 { wt * t.pad[i] } else { 0.0 }) + (if wa > 0.0 { wa * a.pad[i] } else { 0.0 })
        };
        f.pad.push(pad);
        f.weight_tls.push(wt);
        f.weight_als.push(wa);
    }
    // From the top down, while the ALS carries at least half the weight.
    let mut split = f.height.last().map(|h| h + bin_size).unwrap_or(0.0);
    for i in (0..n).rev() {
        if f.weight_als[i].is_nan() {
            continue;
        }
        if f.weight_als[i] >= 0.5 {
            split = f.height[i];
        } else {
            break;
        }
    }
    f.split_height = split;
    f.tls = t;
    f.als = a;
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_splits_at_the_height() {
        let pts = vec![[0.0, 0.0, 0.0]; 4];
        let h = [0.5, 2.0, 3.0, f64::NAN];
        assert_eq!(select(&pts, &h, &Split::Height(2.0), true).unwrap(), [true, false, false, false]);
        assert_eq!(select(&pts, &h, &Split::Height(2.0), false).unwrap(), [false, true, true, false]);
        let r = Raster { data: vec![1.0, f64::NAN], nrows: 1, ncols: 2, xmin: -1.0, ymin: -1.0, resolution: 2.0 };
        let pts2 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 9.0, 0.0]];
        let s = Split::Raster { raster: r, fallback: 5.0 };
        assert_eq!(select(&pts2, &[2.0, 2.0, 2.0], &s, true).unwrap(), [false, true, true]);
        assert!(select(&pts, &h[..2], &Split::Height(1.0), true).is_err());
        assert!(select(&pts, &h, &Split::Height(f64::NAN), true).is_err());
    }

    /// A 2 x 1 x 4 column grid over flat ground at 0.
    fn grid() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // nz = 4 layers of one column pair (ny = 1, nx = 2).
        let beams = vec![100.0, 50.0, 80.0, 40.0, 10.0, 0.0, 2.0, 0.0];
        let pad = vec![0.1, 0.3, 0.2, 0.2, 0.5, f64::NAN, 0.4, f64::NAN];
        let hits = vec![5.0, 7.5, 8.0, 4.0, 2.5, 0.0, 0.4, 0.0];
        let path = vec![100.0, 50.0, 80.0, 40.0, 10.0, 0.0, 2.0, 0.0];
        (beams, pad, hits, path)
    }

    #[test]
    fn layer_statistics_by_height() {
        let (beams, pad, hits, path) = grid();
        let v = VoxelFields { origin: [0.0, 0.0, 0.0], voxel_size: 1.0, shape: [4, 1, 2], beams: &beams, pad: Some(&pad), hits: None, path: None };
        let s = layer_stats(&v, None, None, 1.0, 1.0, 0.5).unwrap();
        assert_eq!(s.height, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s.beams, [75.0, 60.0, 5.0, 1.0]);
        assert_eq!(s.observed, [1.0, 1.0, 0.5, 0.5]);
        assert!((s.pad[0] - 0.2).abs() < 1e-12 && (s.pad[2] - 0.5).abs() < 1e-12);
        // Pooled: sum of hits over sum of path, over G.
        let v2 = VoxelFields { pad: None, hits: Some(&hits), path: Some(&path), ..v };
        let s2 = layer_stats(&v2, None, None, 1.0, 1.0, 0.5).unwrap();
        assert!((s2.pad[0] - 2.0 * 12.5 / 150.0).abs() < 1e-12);
        // Two-metre bins and a mask on the first column.
        let s3 = layer_stats(&v2, Some(&[true, false]), None, 2.0, 1.0, 0.5).unwrap();
        assert_eq!(s3.n_voxels, [2, 2]);
        assert!((s3.pad[0] - 2.0 * 13.0 / 180.0).abs() < 1e-12);
        let dtm = Raster { data: vec![1.0, 1.0], nrows: 1, ncols: 2, xmin: 0.0, ymin: 0.0, resolution: 1.0 };
        let s4 = layer_stats(&v2, None, Some(&dtm), 1.0, 1.0, 0.5).unwrap();
        assert_eq!(s4.height.len(), 3);
        assert!(layer_stats(&VoxelFields { pad: None, hits: None, path: None, ..v2 }, None, None, 1.0, 1.0, 0.5).is_err());
        assert!(layer_stats(&v2, Some(&[true]), None, 1.0, 1.0, 0.5).is_err());
    }

    fn stats(pad: &[f64], beams: &[f64], observed: &[f64]) -> LayerStats {
        LayerStats { height: (0..pad.len()).map(|i| i as f64).collect(), pad: pad.to_vec(), beams: beams.to_vec(), observed: observed.to_vec(), n_voxels: vec![1; pad.len()] }
    }

    #[test]
    fn fusion_weights() {
        let t = stats(&[0.2, 0.3, 0.4, f64::NAN], &[300.0, 100.0, 10.0, 0.0], &[1.0, 1.0, 0.5, 0.0]);
        let a = stats(&[0.1, 0.2, 0.5, 0.6, 0.1], &[5.0, 20.0, 30.0, 40.0, 40.0], &[1.0, 1.0, 1.0, 1.0, 1.0]);
        let f = fuse(&t, &a, 1.0, FuseMode::Beams, 0.0).unwrap();
        assert_eq!(f.height.len(), 5);
        let w0 = 300.0 / 305.0;
        assert!((f.weight_tls[0] - w0).abs() < 1e-12 && (f.pad[0] - (w0 * 0.2 + (1.0 - w0) * 0.1)).abs() < 1e-12);
        assert_eq!((f.weight_tls[3], f.pad[3]), (0.0, 0.6));
        assert_eq!(f.weight_als[4], 1.0);
        assert_eq!(f.split_height, 2.0);
        let b = fuse(&t, &a, 1.0, FuseMode::Best, 0.0).unwrap();
        assert_eq!(b.pad[..3], [0.2, 0.3, 0.5]);
        let o = fuse(&t, &a, 1.0, FuseMode::Observed, 0.6).unwrap();
        assert_eq!((o.weight_tls[2], o.weight_als[2]), (0.0, 1.0));
        assert!((o.weight_tls[0] - 0.5).abs() < 1e-12);
        assert!(fuse(&t, &a, 1.0, FuseMode::Beams, 2.0).is_err());
        assert!(FuseMode::parse("mean").is_err());
    }
}
