// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Point-cloud quality from stems.
//!
//! A tree stem between 1 and 3 m is the one surface in a forest scan whose
//! shape is known well enough to measure a scanner against. Every stem is cut
//! into thin horizontal slices; a circle is fitted to each slice from all its
//! points (all scans together), and every point's radial residual is read
//! against it. Then, per scan position:
//!
//! * the residuals' spread about their own median in the slice (`within`):
//!   range noise, bark and the stem's departure from a circle over the arc the
//!   scan saw;
//! * the spread left after removing a smooth curve along the arc (constant,
//!   first and second harmonics of the angle, `local`): range noise and bark
//!   roughness only -- the stem's shape and the scan's offset are gone, which
//!   takes this under the floor that a whole-stem circle cannot get below;
//! * a horizontal offset `t` minimising `sum (r_p - t . n_p)^2` over every
//!   point of the scan in every slice, `n_p` the outward normal: a rigid
//!   misregistration moves the scan's side of each stem by `t . n`, while
//!   stem shapes average out over many stems.
//!
//! The circle is fitted to all scans, so a misregistered scan pulls it by
//! about its share of the slice's points and its offset reads that much small.
//! The vertical component of a misregistration does not show on vertical
//! stems.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::filters::Rng;
use crate::stems::{angular_coverage, fit_circle_refined, ransac_circle, StemParams};
use crate::Point;

#[derive(Debug, Clone)]
pub struct NoiseParams {
    /// Slice heights (m above ground): from, to, step, and slice thickness.
    pub height_min: f64,
    pub height_max: f64,
    pub step: f64,
    pub thickness: f64,
    /// Accepted slice radii (m).
    pub min_radius: f64,
    pub max_radius: f64,
    /// A slice's circle must see at least this arc (deg) and explain this
    /// share of the points near it.
    pub min_arc: f64,
    pub min_inlier_fraction: f64,
    /// Points further than `max(cut_min, cut_fraction * r)` from the circle
    /// are not stem.
    pub cut_min: f64,
    pub cut_fraction: f64,
    /// Fewest points of one scan in a slice for its per-scan statistics.
    pub min_scan_points: usize,
}

impl Default for NoiseParams {
    fn default() -> Self {
        NoiseParams { height_min: 1.0, height_max: 3.0, step: 0.25, thickness: 0.1, min_radius: 0.05, max_radius: 1.0, min_arc: 270.0, min_inlier_fraction: 0.5, cut_min: 0.05, cut_fraction: 0.3, min_scan_points: 30 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StemSlice {
    pub stem: usize,
    pub height: f64,
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    pub n_points: usize,
    /// Robust spread (1.4826 MAD) of all points' residuals about the circle,
    /// after the scans were moved by their estimated offsets, and before
    /// (the cloud as it stands).
    pub sigma: f64,
    pub sigma_first: f64,
    pub arc: f64,
    /// Share of stem points more than 4 sigma off the circle.
    pub tail_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScanSlice {
    pub scan: i64,
    pub slice: usize,
    pub n_points: usize,
    /// Median residual of the scan's points in the slice (m).
    pub median_residual: f64,
    pub sigma_within: f64,
    pub sigma_local: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScanOffset {
    pub scan: i64,
    pub n_points: usize,
    pub n_slices: usize,
    /// Horizontal misregistration (m).
    pub tx: f64,
    pub ty: f64,
    /// Median over its slices, weighted by points.
    pub sigma_within: f64,
    pub sigma_local: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StemNoise {
    pub slices: Vec<StemSlice>,
    pub scan_slices: Vec<ScanSlice>,
    pub scans: Vec<ScanOffset>,
    /// Residual per input point (NaN where it was not used).
    pub residual: Vec<f64>,
}

fn robust_sigma(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = v[v.len() / 2];
    let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    1.4826 * dev[dev.len() / 2]
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Least squares of `y` on the columns of `x` (small dense problem).
fn lstsq(x: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let k = x.first()?.len();
    let mut a = nalgebra::DMatrix::<f64>::zeros(k, k);
    let mut b = nalgebra::DVector::<f64>::zeros(k);
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..k {
            b[i] += row[i] * yi;
            for j in 0..k {
                a[(i, j)] += row[i] * row[j];
            }
        }
    }
    a.lu().solve(&b).map(|s| s.iter().copied().collect())
}

/// Stem-based noise and registration of a point cloud. `heights` are above
/// ground; `scan_ids` (per point, `None` for a single scan) say which scan
/// position each point came from; `stems` are `(x, y)` stem positions.
///
/// With several scans the offsets are refined `iterations` times: each scan
/// is moved back by its estimate and the circles refitted, which removes the
/// pull of a misregistered scan on the shared circle. Offsets, spreads and
/// residuals are those of the last pass; `sigma_first` keeps each slice's
/// spread before any correction, i.e. as the cloud stands.
pub fn stem_noise(points: &[Point], heights: &[f64], scan_ids: Option<&[i64]>, stems: &[[f64; 2]], p: &NoiseParams, iterations: usize) -> StemNoise {
    let mut shift: HashMap<i64, [f64; 2]> = HashMap::new();
    let mut moved: Vec<Point> = points.to_vec();
    let mut first: Option<HashMap<(usize, i64), f64>> = None;
    let key = |sl: &StemSlice| (sl.stem, (sl.height * 1000.0).round() as i64);
    let mut result = StemNoise::default();
    for pass in 0..iterations.max(1) {
        result = stem_noise_pass(&moved, heights, scan_ids, stems, p);
        if first.is_none() {
            first = Some(result.slices.iter().map(|sl| (key(sl), sl.sigma)).collect());
        }
        let Some(ids) = scan_ids else { break };
        for sc in &result.scans {
            let e = shift.entry(sc.scan).or_insert([0.0, 0.0]);
            e[0] += sc.tx;
            e[1] += sc.ty;
        }
        if pass + 1 == iterations.max(1) {
            break;
        }
        for (i, q) in moved.iter_mut().enumerate() {
            if let Some(t) = shift.get(&ids[i]) {
                q[0] = points[i][0] - t[0];
                q[1] = points[i][1] - t[1];
            }
        }
    }
    for sc in result.scans.iter_mut() {
        if let Some(t) = shift.get(&sc.scan) {
            sc.tx = t[0];
            sc.ty = t[1];
        }
    }
    // Moving every scan alike changes nothing, so only offsets between scans
    // are measured: report them about their point-weighted mean.
    let w: f64 = result.scans.iter().map(|s| s.n_points as f64).sum();
    if result.scans.len() > 1 && w > 0.0 {
        let mx = result.scans.iter().map(|s| s.tx * s.n_points as f64).sum::<f64>() / w;
        let my = result.scans.iter().map(|s| s.ty * s.n_points as f64).sum::<f64>() / w;
        for sc in result.scans.iter_mut() {
            sc.tx -= mx;
            sc.ty -= my;
        }
    }
    // Slices can differ between passes: match the first pass by stem and height.
    let firsts = first.unwrap_or_default();
    for sl in result.slices.iter_mut() {
        sl.sigma_first = firsts.get(&key(sl)).copied().unwrap_or(f64::NAN);
    }
    result
}

fn stem_noise_pass(points: &[Point], heights: &[f64], scan_ids: Option<&[i64]>, stems: &[[f64; 2]], p: &NoiseParams) -> StemNoise {
    let n = points.len();
    let mut out = StemNoise { residual: vec![f64::NAN; n], ..Default::default() };
    if n == 0 || stems.is_empty() || !(p.step > 0.0) || !(p.thickness > 0.0) {
        return out;
    }
    // 1 m grid over the points in the height band.
    let band: Vec<usize> = (0..n).filter(|&i| heights[i] >= p.height_min - p.thickness && heights[i] <= p.height_max + p.thickness).collect();
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for &i in &band {
        grid.entry((points[i][0].floor() as i64, points[i][1].floor() as i64)).or_default().push(i);
    }
    let search = p.max_radius * 1.5 + 0.2;
    let n_levels = (((p.height_max - p.height_min) / p.step).floor() as usize) + 1;
    let scan_of = |i: usize| scan_ids.map_or(0, |s| s[i]);

    // Per stem: its candidate points, then each slice.
    let per_stem: Vec<(Vec<StemSlice>, Vec<(usize, usize, f64, [f64; 2])>)> = stems
        .par_iter()
        .enumerate()
        .map(|(si, st)| {
            let r = search.ceil() as i64;
            let (gx, gy) = (st[0].floor() as i64, st[1].floor() as i64);
            let mut cand: Vec<usize> = Vec::new();
            for dx in -r..=r {
                for dy in -r..=r {
                    if let Some(v) = grid.get(&(gx + dx, gy + dy)) {
                        cand.extend(v.iter().copied().filter(|&i| (points[i][0] - st[0]).hypot(points[i][1] - st[1]) <= search));
                    }
                }
            }
            let cp = StemParams { min_radius: p.min_radius, max_radius: p.max_radius, ransac_tolerance: 0.01, ransac_iterations: 200, ..Default::default() };
            let mut rng = Rng::new(si as u64 + 7);
            let mut slices = Vec::new();
            // (point, slice of this stem, residual, outward normal)
            let mut used: Vec<(usize, usize, f64, [f64; 2])> = Vec::new();
            for lvl in 0..n_levels {
                let h = p.height_min + lvl as f64 * p.step;
                let idx: Vec<usize> = cand.iter().copied().filter(|&i| (heights[i] - h).abs() <= 0.5 * p.thickness).collect();
                if idx.len() < 20 {
                    continue;
                }
                let xy: Vec<[f64; 2]> = idx.iter().map(|&i| [points[i][0], points[i][1]]).collect();
                let Some((cx0, cy0, r0, _)) = ransac_circle(&xy, &cp, &mut rng) else { continue };
                let band_w = 0.02f64.max(0.1 * r0);
                let inl: Vec<[f64; 2]> = xy.iter().copied().filter(|q| ((q[0] - cx0).hypot(q[1] - cy0) - r0).abs() <= band_w).collect();
                let Some((cx, cy, rad)) = fit_circle_refined(&inl) else { continue };
                if !(rad >= p.min_radius && rad <= p.max_radius) {
                    continue;
                }
                let near: Vec<usize> = idx.iter().copied().filter(|&i| (points[i][0] - cx).hypot(points[i][1] - cy) <= 1.3 * rad + 0.1).collect();
                let inl2: Vec<[f64; 2]> = near.iter().map(|&i| [points[i][0], points[i][1]]).filter(|q| ((q[0] - cx).hypot(q[1] - cy) - rad).abs() <= band_w).collect();
                let (_, arc) = angular_coverage(&inl2, cx, cy);
                if arc < p.min_arc || (inl2.len() as f64) < p.min_inlier_fraction * near.len() as f64 {
                    continue;
                }
                let cut = p.cut_min.max(p.cut_fraction * rad);
                let mut res: Vec<(usize, f64, [f64; 2])> = Vec::new();
                for &i in &near {
                    let (dx, dy) = (points[i][0] - cx, points[i][1] - cy);
                    let d = dx.hypot(dy);
                    let rr = d - rad;
                    if rr.abs() <= cut && d > 0.0 {
                        res.push((i, rr, [dx / d, dy / d]));
                    }
                }
                let mut v: Vec<f64> = res.iter().map(|x| x.1).collect();
                let sigma = robust_sigma(&mut v);
                let tail = res.iter().filter(|x| x.1.abs() > 4.0 * sigma).count() as f64 / res.len().max(1) as f64;
                let slice_no = slices.len();
                slices.push(StemSlice { stem: si, height: h, cx, cy, radius: rad, n_points: res.len(), sigma, sigma_first: sigma, arc, tail_fraction: tail });
                used.extend(res.into_iter().map(|(i, rr, nrm)| (i, slice_no, rr, nrm)));
            }
            (slices, used)
        })
        .collect();

    // Flatten, recording each point's slice.
    let mut slice_of_point: Vec<(usize, usize, f64, [f64; 2])> = Vec::new(); // (point, global slice, residual, normal)
    for (slices, used) in per_stem {
        let base = out.slices.len();
        slice_of_point.extend(used.into_iter().map(|(i, s, rr, nrm)| (i, base + s, rr, nrm)));
        out.slices.extend(slices);
    }
    for &(i, _, rr, _) in &slice_of_point {
        out.residual[i] = rr;
    }

    // Per scan and slice.
    let mut groups: HashMap<(i64, usize), Vec<(f64, [f64; 2])>> = HashMap::new();
    for &(i, s, rr, nrm) in &slice_of_point {
        groups.entry((scan_of(i), s)).or_default().push((rr, nrm));
    }
    let mut keys: Vec<(i64, usize)> = groups.keys().copied().collect();
    keys.sort_unstable();
    for (scan, s) in keys {
        let g = &groups[&(scan, s)];
        if g.len() < p.min_scan_points {
            continue;
        }
        let r: Vec<f64> = g.iter().map(|x| x.0).collect();
        let med = median(&mut r.clone());
        let mut dm: Vec<f64> = r.iter().map(|x| x - med).collect();
        let sigma_within = robust_sigma(&mut dm);
        // Smooth curve along the arc: 1, cos a, sin a, cos 2a, sin 2a.
        let rows: Vec<Vec<f64>> = g.iter().map(|x| {
            let a = x.1[1].atan2(x.1[0]);
            vec![1.0, a.cos(), a.sin(), (2.0 * a).cos(), (2.0 * a).sin()]
        }).collect();
        let sigma_local = match lstsq(&rows, &r) {
            Some(c) => {
                let mut left: Vec<f64> = rows.iter().zip(&r).map(|(x, y)| y - x.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>()).collect();
                robust_sigma(&mut left)
            }
            None => f64::NAN,
        };
        out.scan_slices.push(ScanSlice { scan, slice: s, n_points: g.len(), median_residual: med, sigma_within, sigma_local });
    }

    // Per scan: horizontal offset, Huber-weighted (Huber 1964; c = 1.345).
    let mut by_scan: HashMap<i64, Vec<(f64, [f64; 2])>> = HashMap::new();
    for &(i, _, rr, nrm) in &slice_of_point {
        by_scan.entry(scan_of(i)).or_default().push((rr, nrm));
    }
    let mut scans: Vec<i64> = by_scan.keys().copied().collect();
    scans.sort_unstable();
    for scan in scans {
        let pts = &by_scan[&scan];
        let ss: Vec<&ScanSlice> = out.scan_slices.iter().filter(|x| x.scan == scan).collect();
        let wmed = |f: &dyn Fn(&ScanSlice) -> f64| -> f64 {
            let mut v: Vec<(f64, usize)> = ss.iter().map(|x| (f(x), x.n_points)).filter(|x| x.0.is_finite()).collect();
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let total: usize = v.iter().map(|x| x.1).sum();
            let mut acc = 0;
            for (x, w) in &v {
                acc += w;
                if 2 * acc >= total {
                    return *x;
                }
            }
            v[v.len() - 1].0
        };
        let (mut tx, mut ty) = (0.0, 0.0);
        let mut scale = {
            let mut v: Vec<f64> = pts.iter().map(|x| x.0).collect();
            robust_sigma(&mut v).max(1e-4)
        };
        for _ in 0..10 {
            let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (rr, nrm) in pts {
                let e = rr - (tx * nrm[0] + ty * nrm[1]);
                let u = e.abs() / (1.345 * scale);
                let w = if u <= 1.0 { 1.0 } else { 1.0 / u };
                a11 += w * nrm[0] * nrm[0];
                a12 += w * nrm[0] * nrm[1];
                a22 += w * nrm[1] * nrm[1];
                b1 += w * nrm[0] * rr;
                b2 += w * nrm[1] * rr;
            }
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-12 {
                break;
            }
            tx = (a22 * b1 - a12 * b2) / det;
            ty = (a11 * b2 - a12 * b1) / det;
            let mut e: Vec<f64> = pts.iter().map(|(rr, nrm)| rr - (tx * nrm[0] + ty * nrm[1])).collect();
            scale = robust_sigma(&mut e).max(1e-4);
        }
        out.scans.push(ScanOffset { scan, n_points: pts.len(), n_slices: ss.len(), tx, ty, sigma_within: wmed(&|x| x.sigma_within), sigma_local: wmed(&|x| x.sigma_local) });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(rng: &mut Rng) -> f64 {
        let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let v = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    #[test]
    fn recovers_noise_and_offset() {
        // Four scans around a 0.2 m stem at the origin, each seeing the half
        // facing it; 3 mm radial noise; scan 1 shifted 2 cm in x.
        let mut rng = Rng::new(3);
        let (mut pts, mut ids) = (Vec::new(), Vec::new());
        for (scan, facing) in [0.0f64, 90.0, 180.0, 270.0].iter().enumerate() {
            for k in 0..20_000 {
                let a = facing.to_radians() + (k as f64 / 20_000.0 - 0.5) * std::f64::consts::PI;
                let r = 0.2 + 0.003 * gauss(&mut rng);
                let z = 1.0 + 2.0 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let shift = if scan == 1 { 0.02 } else { 0.0 };
                pts.push([r * a.cos() + shift, r * a.sin(), z]);
                ids.push(scan as i64);
            }
        }
        let h: Vec<f64> = pts.iter().map(|q| q[2]).collect();
        let res = stem_noise(&pts, &h, Some(&ids), &[[0.0, 0.0]], &NoiseParams::default(), 1);
        assert!(!res.slices.is_empty());
        let s1 = res.scans.iter().find(|s| s.scan == 1).unwrap();
        let s0 = res.scans.iter().find(|s| s.scan == 0).unwrap();
        assert!((s1.tx - 0.015).abs() < 0.006, "tx {}", s1.tx); // pulled by the shared fit: ~3/4 of 2 cm
        assert!(s1.ty.abs() < 0.004 && s0.tx.abs() < 0.008);
        for s in &res.scans {
            assert!((s.sigma_local - 0.003).abs() < 0.001, "local {}", s.sigma_local);
        }
        // Refitting with the scans moved back recovers the full shift.
        let it = stem_noise(&pts, &h, Some(&ids), &[[0.0, 0.0]], &NoiseParams::default(), 5);
        let s1 = it.scans.iter().find(|s| s.scan == 1).unwrap();
        // Relative to the mean of the four: +1.5 cm for scan 1, -0.5 cm for the others.
        assert!((s1.tx - 0.015).abs() < 0.003, "tx {}", s1.tx);
        assert!(it.scans.iter().filter(|s| s.scan != 1).all(|s| (s.tx + 0.005).abs() < 0.003));
        assert!(it.slices.iter().all(|sl| sl.sigma < sl.sigma_first), "correction must tighten the stems");
    }
}
