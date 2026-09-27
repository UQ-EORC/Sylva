// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Does a stem have buttresses, and how high do they reach?
//!
//! Two signals: a circle stops explaining the bark of the base (RANSAC
//! circles per slice), and protrusions well outside the stem radius persist
//! at the same angle from slice to slice (flanges), where clutter does not.
//! [`crate::qsm::buttress`] rebuilds the base once it is known to be
//! buttressed.

use crate::error::{Error, Result};
use crate::filters::{estimate_normals, planarity_linearity, voxel_downsample_indices};
use crate::numeric::{arange, median, nanmedian, quantile};
use crate::trees::{fit_circle_ransac, RansacCircleParams};
use crate::Point;

/// Settings of [`detect_buttress`]; the defaults are the Python package's.
#[derive(Debug, Clone)]
pub struct ButtressParams {
    /// Horizontal reach from the stem centre (m).
    pub max_radius: f64,
    /// Slice thickness (m).
    pub slice_height: f64,
    /// Highest slice (m); also the highest possible top.
    pub max_height: f64,
    /// Top of the base zone the decision looks at (m).
    pub low: f64,
    /// Angular bins for the ridges.
    pub bins: usize,
    /// A base whose median circle fit is under this may be buttressed.
    pub max_circle_fit: f64,
    /// Ridges needed to call a stem buttressed.
    pub min_ridges: usize,
    /// Use only locally planar, near-vertical surface points.
    pub bark_only: bool,
    /// Thin the points to this spacing first (m); 0 keeps every point.
    pub voxel: f64,
}

impl Default for ButtressParams {
    fn default() -> Self {
        ButtressParams { max_radius: 4.0, slice_height: 0.1, max_height: 6.0, low: 1.0, bins: 36, max_circle_fit: 0.55, min_ridges: 2, bark_only: true, voxel: 0.02 }
    }
}

/// Result of [`detect_buttress`].
#[derive(Debug, Clone, PartialEq)]
pub struct Buttress {
    pub buttressed: bool,
    /// Median share of bark points a circle explains below `low`.
    pub base_circle_fit: f64,
    /// The same above 2 m.
    pub stem_circle_fit: f64,
    /// Stem radius (m) from the round slices above 2 m.
    pub stem_radius: f64,
    pub ridges: usize,
    /// Share of the angle with persistent protrusions.
    pub ridge_share: f64,
    /// 95th percentile distance of the base points from the centre, in stem radii.
    pub spread: f64,
    /// Height (m) where a circle explains the stem again; NaN if not buttressed.
    pub top: f64,
    pub centre: [f64; 2],
}

/// Flanges around the stem: runs of angle where `persistence >= level`,
/// split where persistence dips by `dip` between two peaks (neighbouring
/// flanges). The angle wraps around.
pub fn count_ridges(persistence: &[f64], level: f64, dip: f64) -> usize {
    let ridge: Vec<bool> = persistence.iter().map(|&v| v >= level).collect();
    if !ridge.iter().any(|&r| r) {
        return 0;
    }
    let Some(start) = ridge.iter().position(|&r| !r) else { return 1 };
    // Rotate so that no run wraps around.
    let p: Vec<f64> = persistence[start..].iter().chain(&persistence[..start]).copied().collect();
    let n = p.len();
    let (mut count, mut i) = (0, 0);
    while i < n {
        if p[i] < level {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && p[j] >= level {
            j += 1;
        }
        count += 1;
        let (mut peak, mut low) = (p[i], p[i]);
        for &v in &p[i + 1..j] {
            if v > peak {
                peak = v;
            }
            if v < low {
                low = v;
            }
            if peak - low >= dip && v - low >= dip {
                // A dip between two peaks.
                count += 1;
                peak = v;
                low = v;
            }
        }
        i = j;
    }
    count
}

/// `np.median` of a column: NaN if any value is NaN or there are none.
fn median_nan(v: &[f64]) -> f64 {
    if v.iter().any(|x| x.is_nan()) {
        f64::NAN
    } else {
        median(v)
    }
}

/// Detect a buttressed base from a tree's points and their heights above
/// ground.
///
/// Only points within `max_radius` of the stem centre are used (the median
/// of the points 2.5-3.5 m up, or 1.2-1.8 m if those are under 50, when
/// `base_xy` is `None`). With `bark_only`, only points with planarity >= 0.4
/// over 20 neighbours and a normal within 60 degrees of horizontal take
/// part. A RANSAC circle is fitted to every slice from 0.2 m to
/// `max_height`; bins of angle holding points beyond `1.4 r + 0.1` m in at
/// least 60 % of the slices below `low` are ridges. The stem is buttressed
/// when the median circle fit below `low` is under `max_circle_fit` and
/// there are at least `min_ridges` ridges; its top is then the lowest slice
/// from `low` up that starts three slices a circle explains again.
pub fn detect_buttress(points: &[Point], heights: &[f64], base_xy: Option<[f64; 2]>, p: &ButtressParams) -> Result<Buttress> {
    if heights.len() != points.len() {
        return Err(Error::invalid("heights must have one value per point"));
    }
    if p.bins == 0 {
        return Err(Error::invalid("bins must be positive"));
    }
    let (xyz, h_all): (Vec<Point>, Vec<f64>) = if p.voxel != 0.0 {
        voxel_downsample_indices(points, p.voxel).into_iter().map(|i| (points[i], heights[i])).unzip()
    } else {
        (points.to_vec(), heights.to_vec())
    };
    let centre = match base_xy {
        Some(c) => c,
        None => {
            let band = |lo: f64, hi: f64| -> Vec<usize> { (0..xyz.len()).filter(|&i| h_all[i] > lo && h_all[i] < hi).collect() };
            let mut sel = band(2.5, 3.5);
            if sel.len() < 50 {
                sel = band(1.2, 1.8);
            }
            if sel.is_empty() {
                sel = (0..xyz.len()).collect();
            }
            let col = |k: usize| median_nan(&sel.iter().map(|&i| xyz[i][k]).collect::<Vec<_>>());
            [col(0), col(1)]
        }
    };
    let (mut pts, mut h): (Vec<Point>, Vec<f64>) = xyz
        .iter()
        .zip(&h_all)
        .filter(|(q, &hh)| (q[0] - centre[0]).hypot(q[1] - centre[1]) < p.max_radius && hh >= 0.0 && hh < p.max_height + p.slice_height)
        .map(|(q, &hh)| (*q, hh))
        .unzip();
    if p.bark_only && pts.len() > 20 {
        let (planarity, _) = planarity_linearity(&pts, 20);
        let normals = estimate_normals(&pts, 20);
        let bark: Vec<bool> = planarity.iter().zip(&normals).map(|(&pl, n)| pl >= 0.4 && n[2].abs() <= 0.5).collect();
        let mut k = 0;
        pts.retain(|_| {
            k += 1;
            bark[k - 1]
        });
        let mut k = 0;
        h.retain(|_| {
            k += 1;
            bark[k - 1]
        });
    }
    let z0s = arange(0.2, p.max_height, p.slice_height);
    let mut radius = vec![f64::NAN; z0s.len()];
    let mut fit = vec![f64::NAN; z0s.len()];
    let mut slices: Vec<Vec<[f64; 2]>> = Vec::with_capacity(z0s.len());
    for (k, &z0) in z0s.iter().enumerate() {
        let s: Vec<[f64; 2]> = pts.iter().zip(&h).filter(|(_, &hh)| hh >= z0 && hh < z0 + p.slice_height).map(|(q, _)| [q[0], q[1]]).collect();
        if s.len() >= 30 {
            let rp = RansacCircleParams { threshold: 0.02, iterations: 200, max_radius: p.max_radius, seed: k as u64, ..Default::default() };
            match fit_circle_ransac(&s, &rp) {
                Ok((_, _, r, inl)) => {
                    radius[k] = r;
                    fit[k] = inl.iter().filter(|&&b| b).count() as f64 / inl.len() as f64;
                }
                Err(_) => fit[k] = 0.0,
            }
        }
        slices.push(s);
    }
    let up: Vec<bool> = (0..z0s.len()).map(|k| z0s[k] >= 2.0 && fit[k].is_finite()).collect();
    let round_up: Vec<bool> = (0..z0s.len()).map(|k| up[k] && fit[k] >= 0.5).collect();
    let reference = if round_up.iter().filter(|&&b| b).count() >= 3 { &round_up } else { &up };
    let pick = |v: &[f64], mask: &[bool]| -> f64 {
        if mask.iter().any(|&b| b) {
            nanmedian(v.iter().zip(mask).filter(|(_, &m)| m).map(|(x, _)| *x))
        } else {
            f64::NAN
        }
    };
    let stem_r = pick(&radius, reference);
    let stem_fit = pick(&fit, reference);
    let base: Vec<bool> = (0..z0s.len()).map(|k| z0s[k] < p.low && fit[k].is_finite()).collect();
    let base_fit = pick(&fit, &base);
    let mut marks = 0usize;
    let mut hits = vec![0usize; p.bins];
    let mut spread = Vec::new();
    for (&z0, s) in z0s.iter().zip(&slices) {
        if z0 >= p.low || s.len() < 30 || !stem_r.is_finite() {
            continue;
        }
        let d: Vec<[f64; 2]> = s.iter().map(|q| [q[0] - centre[0], q[1] - centre[1]]).collect();
        let dist: Vec<f64> = d.iter().map(|v| v[0].hypot(v[1])).collect();
        spread.push(quantile(&dist, 0.95));
        let mut mk = vec![false; p.bins];
        let reach = 1.4 * stem_r + 0.1;
        for (v, &r) in d.iter().zip(&dist) {
            if r > reach {
                let b = ((v[1].atan2(v[0]) + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * p.bins as f64) as usize % p.bins;
                mk[b] = true;
            }
        }
        for (c, m) in hits.iter_mut().zip(mk) {
            *c += m as usize;
        }
        marks += 1;
    }
    let persistence: Vec<f64> = if marks > 0 { hits.iter().map(|&c| c as f64 / marks as f64).collect() } else { vec![0.0; p.bins] };
    let ridges = count_ridges(&persistence, 0.6, 0.2);
    let buttressed = base_fit.is_finite() && base_fit < p.max_circle_fit && ridges >= p.min_ridges;
    let mut top = f64::NAN;
    if buttressed && stem_fit.is_finite() {
        // The lowest height from which a circle explains the stem again, for three slices running.
        let need = (0.8 * stem_fit).min(stem_fit - 0.1);
        let good: Vec<bool> = fit.iter().map(|&f| f.is_finite() && f >= need).collect();
        top = p.max_height;
        for k in 0..z0s.len().saturating_sub(2) {
            if z0s[k] >= p.low && good[k..k + 3].iter().all(|&g| g) {
                top = z0s[k];
                break;
            }
        }
    }
    let ridge_share = persistence.iter().filter(|&&v| v >= 0.6).count() as f64 / p.bins as f64;
    let spread = if !spread.is_empty() && stem_r > 0.0 { median(&spread) / stem_r } else { f64::NAN };
    Ok(Buttress { buttressed, base_circle_fit: base_fit, stem_circle_fit: stem_fit, stem_radius: stem_r, ridges, ridge_share, spread, top, centre })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Rng;

    #[test]
    fn ridges_split_at_dips_and_wrap_around() {
        assert_eq!(count_ridges(&[0.0; 36], 0.6, 0.2), 0);
        assert_eq!(count_ridges(&[1.0; 36], 0.6, 0.2), 1);
        // One run across the wrap, one with a dip between two peaks.
        assert_eq!(count_ridges(&[0.9, 0.9, 0.2, 0.9, 0.3, 0.8, 0.95, 0.7, 0.9], 0.6, 0.2), 3);
        assert_eq!(count_ridges(&[0.1, 0.9, 0.65, 0.9, 0.1], 0.6, 0.2), 2);
    }

    /// A 0.25 m stem to 6 m, with five flanges fading out by 2 m if `flanges`.
    fn base(flanges: bool) -> (Vec<Point>, Vec<f64>) {
        let mut rng = Rng::new(3);
        let mut uniform = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let pts: Vec<Point> = (0..60_000)
            .map(|_| {
                let t = uniform() * std::f64::consts::TAU;
                let h = uniform() * 6.0;
                let f = if flanges { 3.0 * (1.0 - h / 2.0).max(0.0) * (2.5 * t).cos().powi(8) } else { 0.0 };
                let r = 0.25 * (1.0 + f);
                [r * t.cos(), r * t.sin(), h]
            })
            .collect();
        let h = pts.iter().map(|q| q[2]).collect();
        (pts, h)
    }

    #[test]
    fn flanged_bases_are_buttressed_and_round_ones_are_not() {
        let p = ButtressParams { bark_only: false, ..Default::default() };
        let (pts, h) = base(true);
        let b = detect_buttress(&pts, &h, Some([0.0, 0.0]), &p).unwrap();
        assert!(b.buttressed && b.ridges >= 2, "{b:?}");
        assert!((1.0..=2.6).contains(&b.top), "{b:?}");
        let (pts, h) = base(false);
        let b = detect_buttress(&pts, &h, None, &p).unwrap();
        assert!(!b.buttressed && (b.stem_radius - 0.25).abs() < 0.03, "{b:?}");
        assert!(b.top.is_nan());
    }
}
