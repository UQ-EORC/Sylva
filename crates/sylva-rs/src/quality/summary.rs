// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Plot-level figures from a stem-noise result ([`crate::quality`]), and
//! scan positions from sensor origins.
//!
//! The summary works on the result's tables as plain columns, so that the
//! result stays data in every language that binds it.

use std::collections::{BTreeSet, HashMap};

use crate::error::{Error, Result};
use crate::util::numeric::pairwise_sum;

/// A float's conversion to int64 as NumPy does it on x86: NaN and values
/// out of range become the smallest int64.
fn to_i64(v: f64) -> i64 {
    if (-9.223372036854776e18..9.223372036854776e18).contains(&v) {
        v as i64
    } else {
        i64::MIN
    }
}

/// Scan position of each point from its sensor origin: origins on the same
/// `tolerance` grid cell (rounded half to even) are one position, numbered
/// 0..n-1 in lexicographic order of the cells. `origins` is row-major with
/// `ncols` values per point.
pub fn scan_ids_from_origins(origins: &[f64], ncols: usize, tolerance: f64) -> Result<Vec<i64>> {
    if ncols == 0 || !origins.len().is_multiple_of(ncols) {
        return Err(Error::invalid("origins must have the same number of values per point"));
    }
    let keys: Vec<Vec<i64>> = origins.chunks(ncols).map(|row| row.iter().map(|&v| to_i64((v / tolerance).round_ties_even())).collect()).collect();
    let unique: BTreeSet<&Vec<i64>> = keys.iter().collect();
    let id: HashMap<&Vec<i64>, i64> = unique.into_iter().enumerate().map(|(i, k)| (k, i as i64)).collect();
    Ok(keys.iter().map(|k| id[k]).collect())
}

/// Weighted median: the smallest finite value, in sorted order, at which
/// the cumulative weight reaches half the total. Values with a weight that
/// is not positive are left out; NaN if none remain.
pub fn weighted_median(x: &[f64], w: &[f64]) -> f64 {
    let mut ok: Vec<(f64, f64)> = x.iter().zip(w).filter(|(v, wt)| v.is_finite() && **wt > 0.0).map(|(v, wt)| (*v, *wt)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut c = Vec::with_capacity(ok.len());
    let mut s = 0.0;
    for (_, wt) in &ok {
        s += wt;
        c.push(s);
    }
    let half = c[c.len() - 1] / 2.0;
    ok[c.partition_point(|&v| v < half).min(ok.len() - 1)].0
}

/// `np.average(a, weights=w)`.
fn average(a: &[f64], w: &[f64]) -> Result<f64> {
    let scl = pairwise_sum(w);
    if scl == 0.0 {
        return Err(Error::invalid("weights sum to zero, can't be normalized"));
    }
    let aw: Vec<f64> = a.iter().zip(w).map(|(x, y)| x * y).collect();
    Ok(pairwise_sum(&aw) / scl)
}

/// The columns of a stem-noise result that its summary reads.
#[derive(Debug, Clone, Copy)]
pub struct NoiseTables<'a> {
    /// Per stem slice.
    pub slice_count: usize,
    pub slice_stem: &'a [i64],
    pub slice_n_points: &'a [f64],
    pub slice_sigma: &'a [f64],
    pub slice_sigma_first: &'a [f64],
    pub slice_tail_fraction: &'a [f64],
    /// Per scan and slice.
    pub scan_slice_n_points: &'a [f64],
    pub scan_slice_sigma_within: &'a [f64],
    pub scan_slice_sigma_local: &'a [f64],
    /// Per scan.
    pub scan: &'a [i64],
    pub scan_n_points: &'a [f64],
    pub scan_n_slices: &'a [f64],
    pub scan_tx: &'a [f64],
    pub scan_ty: &'a [f64],
}

/// Plot-level figures of a stem-noise result (m).
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseSummary {
    pub n_stems: usize,
    pub n_slices: usize,
    pub n_scans: usize,
    /// Present when a slice qualified.
    pub measured: Option<NoiseMeasured>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseMeasured {
    pub sigma_total: f64,
    pub sigma_corrected: f64,
    pub sigma_within: f64,
    pub sigma_local: f64,
    pub tail_fraction: f64,
    pub n_scans_registered: usize,
    /// With more than one registered scan: `(rms, max, worst scan)` of the
    /// horizontal offsets.
    pub registration: Option<(f64, f64, i64)>,
}

/// Point-weighted medians of the slice and scan spreads, the point-weighted
/// tail fraction, and the RMS and largest horizontal scan offsets over scans
/// measured in at least `min_scan_slices` slices.
pub fn noise_summary(t: &NoiseTables<'_>, min_scan_slices: f64) -> Result<NoiseSummary> {
    let n = t.slice_count;
    let n_stems = if n > 0 { t.slice_stem.iter().collect::<BTreeSet<_>>().len() } else { 0 };
    let mut out = NoiseSummary { n_stems, n_slices: n, n_scans: t.scan.len(), measured: None };
    if n == 0 {
        return Ok(out);
    }
    let ok: Vec<usize> = (0..t.scan.len()).filter(|&i| t.scan_n_slices[i] >= min_scan_slices).collect();
    let registration = if ok.len() > 1 {
        let off: Vec<f64> = ok.iter().map(|&i| t.scan_tx[i].hypot(t.scan_ty[i])).collect();
        let w: Vec<f64> = ok.iter().map(|&i| t.scan_n_points[i]).collect();
        let sq: Vec<f64> = off.iter().map(|o| o * o).collect();
        let rms = average(&sq, &w)?.sqrt();
        // np.max and np.argmax: the first NaN wins, else the first largest.
        let k = match off.iter().position(|o| o.is_nan()) {
            Some(k) => k,
            None => (0..off.len()).fold(0, |b, i| if off[i] > off[b] { i } else { b }),
        };
        Some((rms, off[k], t.scan[ok[k]]))
    } else {
        None
    };
    out.measured = Some(NoiseMeasured {
        sigma_total: weighted_median(t.slice_sigma_first, t.slice_n_points),
        sigma_corrected: weighted_median(t.slice_sigma, t.slice_n_points),
        sigma_within: weighted_median(t.scan_slice_sigma_within, t.scan_slice_n_points),
        sigma_local: weighted_median(t.scan_slice_sigma_local, t.scan_slice_n_points),
        tail_fraction: average(t.slice_tail_fraction, t.slice_n_points)?,
        n_scans_registered: ok.len(),
        registration,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_ids_group_origins_by_cell() {
        let o = [0.0, 0.0, 1.5, 0.01, 0.0, 1.5, 10.0, 0.0, 1.6, 10.0, 0.02, 1.6, 0.0, 0.0, 1.51];
        assert_eq!(scan_ids_from_origins(&o, 3, 0.05).unwrap(), vec![0, 0, 1, 1, 0]);
        // Half cells round to even, as NumPy does.
        assert_eq!(scan_ids_from_origins(&[0.5, 1.5, 2.5], 1, 1.0).unwrap(), vec![0, 1, 1]);
        assert!(scan_ids_from_origins(&[1.0, 2.0], 3, 1.0).is_err());
    }

    #[test]
    fn weighted_median_takes_half_the_weight() {
        assert_eq!(weighted_median(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4]), 2.0);
        assert_eq!(weighted_median(&[1.0, 2.0, 3.0], &[1.0, 1.0, 5.0]), 3.0);
        assert_eq!(weighted_median(&[f64::NAN, 5.0, 1.0], &[9.0, 1.0, 0.0]), 5.0);
        assert!(weighted_median(&[f64::NAN, 1.0], &[1.0, 0.0]).is_nan());
    }

    #[test]
    fn summary_skips_unsupported_scans() {
        let t = NoiseTables {
            slice_count: 2,
            slice_stem: &[0, 0],
            slice_n_points: &[100.0, 100.0],
            slice_sigma: &[0.005, 0.005],
            slice_sigma_first: &[0.006, 0.006],
            slice_tail_fraction: &[0.0, 0.0],
            scan_slice_n_points: &[50.0, 50.0],
            scan_slice_sigma_within: &[0.004, 0.004],
            scan_slice_sigma_local: &[0.003, 0.003],
            scan: &[0, 1, 2],
            scan_n_points: &[500.0, 500.0, 5.0],
            scan_n_slices: &[4.0, 4.0, 0.0],
            scan_tx: &[0.002, -0.002, 0.6],
            scan_ty: &[0.0, 0.0, 0.0],
        };
        let s = noise_summary(&t, 1.0).unwrap();
        let m = s.measured.unwrap();
        assert_eq!((s.n_stems, s.n_slices, s.n_scans, m.n_scans_registered), (1, 2, 3, 2));
        assert_eq!(m.registration.unwrap().1, 0.002);
        let loose = noise_summary(&t, 0.0).unwrap().measured.unwrap();
        assert_eq!(loose.registration.unwrap().1, 0.6);
        assert_eq!(loose.registration.unwrap().2, 2);
    }
}
