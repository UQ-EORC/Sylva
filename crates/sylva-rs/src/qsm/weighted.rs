// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! QSMs fitted with a per-point wood weight.
//!
//! A leaf / wood classifier that drops points is a hard decision: a trunk it
//! misses loses its measurements, and the radius then comes from the priors.
//! Here every point (or every point above a small floor) stays in the graph
//! and the skeleton, so connectivity is what it would be without a filter,
//! and the weight in `[0, 1]` (a wood confidence) enters only where a radius
//! is measured: RANSAC scores a candidate circle by the summed weight of its
//! inliers rather than their number, the refit is a weighted least-squares
//! fit, and a section counts as measured only when its inliers carry enough
//! weight. Weights of one everywhere give exactly the unweighted model.

use crate::error::{Error, Result};
use crate::filters::Rng;
use crate::trees::stems::{angular_coverage, StemParams};
use crate::Point;

use super::model::{fit_cylinders_with, fourier_area_radius, power_mean_radius, scaled_to_spacing, skeletonize, Qsm, QsmParams, BINS};

/// Weight at or above which a point counts as confident wood: the pool
/// RANSAC samples from, the points that set a section's arc and its
/// contour, and the points `n_points` counts.
pub const CONFIDENT: f64 = 0.5;

/// Per-point weights must be one per point, finite and in `[0, 1]`.
pub fn check_weights(weights: &[f64], n: usize) -> Result<()> {
    if weights.len() != n {
        return Err(Error::invalid(format!("weights must have one value per point ({} for {n} points)", weights.len())));
    }
    if let Some(i) = weights.iter().position(|w| !(0.0..=1.0).contains(w)) {
        return Err(Error::invalid(format!("weights must be finite and in [0, 1] (point {i} has {})", weights[i])));
    }
    Ok(())
}

/// [`super::build_qsm`] with a wood weight per point.
///
/// Points with a weight below `p.min_weight` are dropped first; the rest build
/// the graph and skeleton, and the weights steer the radius of each section
/// ([`section_fit`]). Errors when the weights do not match the points or are
/// outside `[0, 1]`, or when no point reaches `min_weight`.
pub fn build_qsm_weighted(xyz: &[Point], base_xy: Option<[f64; 2]>, p: &QsmParams, weights: &[f64]) -> Result<Qsm> {
    check_weights(weights, xyz.len())?;
    let (pts, w): (Vec<Point>, Vec<f64>) = xyz.iter().zip(weights).filter(|(_, &w)| w >= p.min_weight).map(|(q, &w)| (*q, w)).unzip();
    if pts.is_empty() {
        return Err(Error::invalid(format!("no point has a weight of at least min_weight = {}", p.min_weight)));
    }
    let task = crate::util::progress::start("building a QSM", 2);
    let p = &scaled_to_spacing(&pts, p);
    let skel = skeletonize(&pts, base_xy, p)?;
    task.inc(1);
    let qsm = fit_cylinders_with(&pts, &skel, p, Some(&w));
    task.inc(1);
    qsm
}

/// Radius of one section from its points `xy` (in the section's plane) and
/// their weights: the weighted counterpart of the circle fit in
/// [`super::fit_cylinders`], returning the same `(radius, points, arc,
/// inlier share)`.
///
/// The section needs a summed weight of `fit_min_points`; the circle is the
/// weighted RANSAC of [`ransac_circle_weighted`] refitted by weighted least
/// squares on the points within the band; its inliers must carry a summed
/// weight of `min_points` and a mean weight of `min_mean_weight`, and the
/// share is the inliers' weight over the section's. The arc, the Fourier
/// contour and the point count use the confident points only (weight at least
/// [`CONFIDENT`]).
pub(crate) fn section_fit(xy: &[[f64; 2]], w: &[f64], seed: u64, p: &QsmParams, cp: &StemParams) -> Option<(f64, usize, f64, f64)> {
    let total_w: f64 = w.iter().sum();
    if total_w < p.fit_min_points as f64 {
        return None;
    }
    let confident: Vec<[f64; 2]> = xy.iter().zip(w).filter(|(_, &x)| x >= CONFIDENT).map(|(q, _)| *q).collect();
    if p.radius_power > 0.0 {
        return power_mean_radius(&confident, p);
    }
    let mut rng = Rng::new(seed);
    let (cx0, cy0, r0, _) = ransac_circle_weighted(xy, w, cp, &mut rng)?;
    let band = p.ransac_threshold.max(p.relative_tolerance * r0);
    let within = |cx: f64, cy: f64, r: f64| -> (Vec<[f64; 2]>, Vec<f64>) { xy.iter().zip(w).filter(|(q, _)| ((q[0] - cx).hypot(q[1] - cy) - r).abs() <= band).map(|(q, &x)| (*q, x)).unzip() };
    let (bxy, bw) = within(cx0, cy0, r0);
    let (cx, cy, r) = fit_circle_refined_weighted(&bxy, &bw).unwrap_or((cx0, cy0, r0));
    let (ixy, iw) = within(cx, cy, r);
    let w_in: f64 = iw.iter().sum();
    if w_in < p.min_points as f64 || w_in < p.min_mean_weight * ixy.len() as f64 {
        return None;
    }
    let rmse = (ixy.iter().zip(&iw).map(|(q, &x)| ((q[0] - cx).hypot(q[1] - cy) - r).powi(2) * x).sum::<f64>() / w_in).sqrt();
    let sure: Vec<[f64; 2]> = ixy.iter().zip(&iw).filter(|(_, &x)| x >= CONFIDENT).map(|(q, _)| *q).collect();
    let n_sure = sure.len();
    let (_, arc) = angular_coverage(&sure, cx, cy);
    let frac = w_in / total_w;
    let ok = arc >= p.min_arc_deg && rmse <= p.max_rmse.max(band) && frac >= p.min_inlier_fraction;
    if ok && frac >= p.buttress_max_inlier_fraction {
        if p.fourier_min_radius > 0.0 && r >= p.fourier_min_radius {
            if let Some(req) = fourier_area_radius(&confident, cx, cy, BINS * 5 / 6) {
                if req >= p.apex_radius && req <= p.max_radius {
                    return Some((req, n_sure, arc, frac));
                }
            }
        }
        return Some((r, n_sure, arc, frac));
    }
    if p.buttress_equivalent_area && !confident.is_empty() {
        let n = confident.len() as f64;
        let (mx, my) = (confident.iter().map(|q| q[0]).sum::<f64>() / n, confident.iter().map(|q| q[1]).sum::<f64>() / n);
        if let Some(req) = fourier_area_radius(&confident, mx, my, BINS * 5 / 6) {
            if req >= p.apex_radius && req <= p.max_radius {
                return Some((req, confident.len().min(p.fit_min_points), 0.0, frac.max(1e-3)));
            }
        }
    }
    ok.then_some((r, n_sure, arc, frac))
}

fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<(f64, f64, f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let (s1, s2, s3) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
    let cx = (s1 * (b[1] - c[1]) + s2 * (c[1] - a[1]) + s3 * (a[1] - b[1])) / d;
    let cy = (s1 * (c[0] - b[0]) + s2 * (a[0] - c[0]) + s3 * (b[0] - a[0])) / d;
    Some((cx, cy, (a[0] - cx).hypot(a[1] - cy)))
}

/// RANSAC circle scored by the summed weight of its inliers: the weighted
/// form of the circle RANSAC in [`crate::trees::stems`] (its score,
/// `weight * (1 - weighted mean residual / tol)`, and its adaptive stop on
/// the share of the total weight explained), with the weighted refit.
/// Candidate triples are drawn from the confident points (weight at least
/// [`CONFIDENT`]) when there are three of them, otherwise from every point
/// with a positive weight. Weights of one give the unweighted result exactly.
/// Returns `(cx, cy, r, inlier mask)`.
pub(crate) fn ransac_circle_weighted(xy: &[[f64; 2]], w: &[f64], p: &StemParams, rng: &mut Rng) -> Option<(f64, f64, f64, Vec<bool>)> {
    let n = xy.len();
    if n < 3 || w.len() != n {
        return None;
    }
    let mut pool: Vec<usize> = (0..n).filter(|&i| w[i] >= CONFIDENT).collect();
    if pool.len() < 3 {
        pool = (0..n).filter(|&i| w[i] > 0.0).collect();
    }
    let m = pool.len();
    if m < 3 {
        return None;
    }
    let total_w: f64 = w.iter().sum();
    let mut best_score = 0.0;
    let mut best_count = 0usize;
    let mut best_w = 0.0;
    let mut best = (0.0, 0.0, 0.0);
    let mut tried = 0usize;
    // When to stop drawing triples: once the best circle explains nearly all
    // the weight, or once enough triples have been tried that the chance of
    // never having drawn three inliers is below 1 in 1000. That is the
    // standard RANSAC estimate, with the inlier *share* measured by weight.
    let stop = |best_count: usize, best_w: f64, tried: usize| {
        if best_count < 3 {
            return false;
        }
        let ratio = best_w / total_w;
        ratio > 0.99 || tried as f64 >= (1e-3f64).ln() / (1.0 - ratio.powi(3)).max(1e-12).ln()
    };
    // Score one candidate circle and keep it if it is the best so far. A
    // circle earns the weight of every point within `ransac_tolerance` of it,
    // less a penalty for how far inside that band those points sit, so a
    // circle that merely passes near many points loses to one they lie on.
    let consider = |cx: f64, cy: f64, r: f64, best_score: &mut f64, best_count: &mut usize, best_w: &mut f64, best: &mut (f64, f64, f64)| {
        let mut count = 0usize;
        let mut sw = 0.0;
        let mut total = 0.0;
        for (q, &x) in xy.iter().zip(w) {
            let res = ((q[0] - cx).hypot(q[1] - cy) - r).abs();
            if res < p.ransac_tolerance {
                count += 1;
                sw += x;
                total += x * res;
            }
        }
        if count < 3 || sw <= 0.0 {
            return;
        }
        let score = sw * (1.0 - total / sw / p.ransac_tolerance);
        if score > *best_score {
            *best_score = score;
            *best_count = count;
            *best_w = sw;
            *best = (cx, cy, r);
        }
    };
    let block = p.ransac_block.max(1);
    if p.ransac_presample {
        let triples: Vec<(usize, usize, usize)> = (0..p.ransac_iterations).map(|_| (rng.below(m), rng.below(m), rng.below(m))).collect();
        let candidates: Vec<(f64, f64, f64)> = triples
            .into_iter()
            .filter(|&(i, j, k)| i != j && j != k && i != k)
            .filter_map(|(i, j, k)| circumcircle(xy[pool[i]], xy[pool[j]], xy[pool[k]]))
            .filter(|&(_, _, r)| r >= p.min_radius && r <= p.max_radius)
            .collect();
        for chunk in candidates.chunks(block) {
            if stop(best_count, best_w, tried) {
                break;
            }
            tried += chunk.len();
            for &(cx, cy, r) in chunk {
                consider(cx, cy, r, &mut best_score, &mut best_count, &mut best_w, &mut best);
            }
        }
    } else {
        for _ in 0..p.ransac_iterations {
            if tried.is_multiple_of(block) && stop(best_count, best_w, tried) {
                break;
            }
            let (i, j, k) = (rng.below(m), rng.below(m), rng.below(m));
            if i == j || j == k || i == k {
                continue;
            }
            tried += 1;
            let Some((cx, cy, r)) = circumcircle(xy[pool[i]], xy[pool[j]], xy[pool[k]]) else { continue };
            if r < p.min_radius || r > p.max_radius {
                continue;
            }
            consider(cx, cy, r, &mut best_score, &mut best_count, &mut best_w, &mut best);
        }
    }
    if best_score <= 0.0 {
        return None;
    }
    let (mut cx, mut cy, mut r) = best;
    let mut mask: Vec<bool> = xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() < p.ransac_tolerance).collect();
    for _ in 0..3 {
        let (pts, pw): (Vec<[f64; 2]>, Vec<f64>) = xy.iter().zip(w).zip(&mask).filter(|(_, &m)| m).map(|((q, &x), _)| (*q, x)).unzip();
        if pts.len() < 3 {
            break;
        }
        let (ncx, ncy, nr) = fit_circle_refined_weighted(&pts, &pw)?;
        if nr < p.min_radius || nr > p.max_radius {
            return None;
        }
        cx = ncx;
        cy = ncy;
        r = nr;
        let new_mask: Vec<bool> = xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() < p.ransac_tolerance).collect();
        let cnt = new_mask.iter().filter(|&&m| m).count();
        if cnt < 3 || new_mask == mask {
            if cnt >= 3 {
                mask = new_mask;
            }
            break;
        }
        mask = new_mask;
    }
    Some((cx, cy, r, mask))
}

/// Weighted Kåsa (1976) fit about the weighted centroid, then weighted
/// Gauss-Newton geometric refinement: minimises `sum w (d - r)^2`. Weights
/// of one give the unweighted fit of [`crate::trees::stems`] exactly.
pub fn fit_circle_refined_weighted(xy: &[[f64; 2]], w: &[f64]) -> Option<(f64, f64, f64)> {
    if xy.len() < 3 || w.len() != xy.len() {
        return None;
    }
    let sw: f64 = w.iter().sum();
    if sw.is_nan() || sw <= 0.0 || w.iter().filter(|&&x| x > 0.0).count() < 3 {
        return None;
    }
    let mx = xy.iter().zip(w).map(|(q, &x)| q[0] * x).sum::<f64>() / sw;
    let my = xy.iter().zip(w).map(|(q, &x)| q[1] * x).sum::<f64>() / sw;
    let mut ata = nalgebra::Matrix3::<f64>::zeros();
    let mut atb = nalgebra::Vector3::<f64>::zeros();
    for (q, &x) in xy.iter().zip(w) {
        let (px, py) = (q[0] - mx, q[1] - my);
        let row = nalgebra::Vector3::new(2.0 * px, 2.0 * py, 1.0);
        ata += (row * row.transpose()) * x;
        atb += row * ((px * px + py * py) * x);
    }
    let s = ata.lu().solve(&atb)?;
    let r2 = s[2] + s[0] * s[0] + s[1] * s[1];
    if !r2.is_finite() || r2 <= 0.0 {
        return None;
    }
    let (mut cx, mut cy, mut r) = (s[0], s[1], r2.sqrt());
    for _ in 0..8 {
        let mut jtj = nalgebra::Matrix3::<f64>::zeros();
        let mut jtr = nalgebra::Vector3::<f64>::zeros();
        for (q, &x) in xy.iter().zip(w) {
            let dx = q[0] - mx - cx;
            let dy = q[1] - my - cy;
            let d = dx.hypot(dy).max(1e-9);
            let res = d - r;
            let j = nalgebra::Vector3::new(-dx / d, -dy / d, -1.0);
            jtj += (j * j.transpose()) * x;
            jtr += j * (res * x);
        }
        let Some(step) = jtj.lu().solve(&(-jtr)) else { break };
        cx += step[0];
        cy += step[1];
        r += step[2];
        if r <= 0.0 {
            return None;
        }
        if step.norm() < 1e-9 {
            break;
        }
    }
    Some((cx + mx, cy + my, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::nprandom::Generator;
    use crate::qsm::{build_qsm, fit_cylinders};

    /// `n` points on a circle of radius `r` about `(cx, cy)` with radial noise.
    fn circle(rng: &mut Generator, cx: f64, cy: f64, r: f64, n: usize, noise: f64) -> Vec<[f64; 2]> {
        (0..n)
            .map(|_| {
                let t = rng.random() * std::f64::consts::TAU;
                let rr = r + noise * (rng.random() - 0.5);
                [cx + rr * t.cos(), cy + rr * t.sin()]
            })
            .collect()
    }

    fn clutter(rng: &mut Generator, half: f64, n: usize) -> Vec<[f64; 2]> {
        (0..n).map(|_| [(rng.random() * 2.0 - 1.0) * half, (rng.random() * 2.0 - 1.0) * half]).collect()
    }

    /// A vertical stem of radius `r` and height `h` (`n` points, 3 mm noise).
    fn stem(rng: &mut Generator, r: f64, h: f64, n: usize) -> Vec<Point> {
        (0..n)
            .map(|_| {
                let t = rng.random() * std::f64::consts::TAU;
                let rr = r + 0.006 * (rng.random() - 0.5);
                [rr * t.cos(), rr * t.sin(), rng.random() * h]
            })
            .collect()
    }

    #[test]
    fn low_weight_clutter_does_not_pull_the_circle() {
        let mut rng = Generator::new(3);
        // A small arc of bark in a dense cloud of foliage: by count the
        // foliage wins, by weight the bark.
        let mut xy = circle(&mut rng, 0.1, -0.05, 0.12, 120, 0.004);
        let n_bark = xy.len();
        xy.extend(clutter(&mut rng, 0.6, 1500));
        let w: Vec<f64> = (0..xy.len()).map(|i| if i < n_bark { 1.0 } else { 0.05 }).collect();
        let p = StemParams { min_radius: 0.01, max_radius: 1.0, ransac_iterations: 120, ransac_tolerance: 0.01, ..Default::default() };
        let (cx, cy, r, mask) = ransac_circle_weighted(&xy, &w, &p, &mut Rng::new(1)).unwrap();
        assert!((cx - 0.1).abs() < 0.01 && (cy + 0.05).abs() < 0.01 && (r - 0.12).abs() < 0.005, "{cx} {cy} {r}");
        assert!(mask[..n_bark].iter().filter(|&&m| m).count() > 100);
        // The refit on the same points ignores what carries no weight at all.
        let w0: Vec<f64> = w.iter().map(|&x| if x < 1.0 { 0.0 } else { 1.0 }).collect();
        let (cx, cy, r) = fit_circle_refined_weighted(&xy, &w0).unwrap();
        assert!((cx - 0.1).abs() < 2e-3 && (cy + 0.05).abs() < 2e-3 && (r - 0.12).abs() < 2e-3, "{cx} {cy} {r}");
        // Too little weight to fit anything.
        assert!(fit_circle_refined_weighted(&xy, &vec![0.0; xy.len()]).is_none());
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_circle_fit() {
        let mut rng = Generator::new(8);
        for trial in 0..20 {
            let mut xy = circle(&mut rng, 0.0, 0.0, 0.05 + 0.02 * trial as f64, 80, 0.01);
            xy.extend(clutter(&mut rng, 0.5, 40 + 10 * trial));
            let ones = vec![1.0; xy.len()];
            assert_eq!(fit_circle_refined_weighted(&xy, &ones), crate::trees::stems::fit_circle_refined(&xy));
            for p in [StemParams { min_radius: 0.0025, max_radius: 1.0, ransac_iterations: 120, ransac_tolerance: 0.02, ..Default::default() }, StemParams::coreg()] {
                let a = ransac_circle_weighted(&xy, &ones, &p, &mut Rng::new(trial as u64 + 1));
                let b = crate::trees::stems::ransac_circle(&xy, &p, &mut Rng::new(trial as u64 + 1));
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_model() {
        let cloud = crate::synthetic::tree(0.0, 0.0, 0.4, 12.0, 0.0, 6, 3000, 5);
        let ones = vec![1.0; cloud.xyz.len()];
        let p = QsmParams::default();
        assert_eq!(build_qsm_weighted(&cloud.xyz, None, &p, &ones).unwrap(), build_qsm(&cloud.xyz, None, &p).unwrap());
        // Also without the spacing scaling and with the power mean.
        for p in [QsmParams { spacing_scale: 0.0, ..Default::default() }, QsmParams { radius_power: 0.25, ..Default::default() }] {
            let skel = skeletonize(&cloud.xyz, None, &p).unwrap();
            assert_eq!(fit_cylinders_with(&cloud.xyz, &skel, &p, Some(&ones)).unwrap(), fit_cylinders(&cloud.xyz, &skel, &p).unwrap());
        }
    }

    #[test]
    fn weights_keep_foliage_off_a_stem() {
        let mut rng = Generator::new(11);
        let (r, h) = (0.12, 5.0);
        let mut pts = stem(&mut rng, r, h, 12000);
        let n_wood = pts.len();
        // Foliage sheathing the stem from 1.5 m up, twice as many points.
        for _ in 0..24000 {
            let t = rng.random() * std::f64::consts::TAU;
            let d = r + 0.02 + 0.3 * rng.random();
            pts.push([d * t.cos(), d * t.sin(), 1.5 + rng.random() * (h - 1.5)]);
        }
        let w: Vec<f64> = (0..pts.len()).map(|i| if i < n_wood { 0.9 } else { 0.1 }).collect();
        let truth = std::f64::consts::PI * r * r * h;
        let p = QsmParams::default();
        let q = build_qsm_weighted(&pts, None, &p, &w).unwrap();
        let err = (q.total_volume() - truth) / truth;
        assert!(err.abs() < 0.15, "weighted volume {} against {truth}", q.total_volume());
        let dbh = q.dbh();
        assert!((dbh - 2.0 * r).abs() < 0.03, "dbh {dbh}");
        // Measured cylinders report their confident inliers.
        assert!(q.cylinders.iter().any(|c| c.n_points > 0));
        // Dropping the foliage by a floor leaves the bare stem.
        let bare = build_qsm_weighted(&pts, None, &QsmParams { min_weight: 0.5, ..Default::default() }, &w).unwrap();
        assert!(((bare.total_volume() - truth) / truth).abs() < 0.15);
        // Foliage alone, all low weight, is not measured anywhere.
        let leaf: Vec<Point> = pts[n_wood..].to_vec();
        let lw = vec![0.1; leaf.len()];
        let q = build_qsm_weighted(&leaf, None, &p, &lw).unwrap();
        assert!(q.cylinders.iter().all(|c| c.n_points == 0));
    }

    #[test]
    fn classifier_scores_leave_the_labels_alone() {
        use crate::leaves::{classify_leaf_wood, classify_leaf_wood_gbs, classify_leaf_wood_gbs_scores, classify_leaf_wood_scores, gbs_confidence, passage_confidence};
        use crate::pointcloud::Attr;
        let cloud = crate::synthetic::tree(0.0, 0.0, 0.4, 12.0, 0.0, 6, 4000, 9);
        let Some(Attr::U8(cls)) = cloud.attrs.get("classification") else { panic!() };
        let wood: Vec<bool> = cls.iter().map(|&c| c == 5).collect();
        let mean = |v: &[f64], want: bool| -> f64 {
            let sel: Vec<f64> = v.iter().zip(&wood).filter(|(_, &w)| w == want).map(|(x, _)| *x).collect();
            sel.iter().sum::<f64>() / sel.len() as f64
        };
        let wp = crate::qsm::wood::WoodParams::default();
        let s = classify_leaf_wood_scores(&cloud.xyz, 0.02, &wp);
        assert_eq!(s.mask, classify_leaf_wood(&cloud.xyz, 0.02, &wp));
        assert!(s.anisotropy.iter().chain(&s.passage).all(|v| (0.0..=1.0).contains(v)));
        let c = passage_confidence(&s);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(mean(&c, true) > mean(&c, false) + 0.2);
        let gp = crate::qsm::wood::GbsParams::default();
        let (mask, votes) = classify_leaf_wood_gbs_scores(&cloud.xyz, 0.02, &gp);
        assert_eq!(mask, classify_leaf_wood_gbs(&cloud.xyz, 0.02, &gp));
        let c = gbs_confidence(&mask, &votes);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(mean(&c, true) > mean(&c, false) + 0.2);
        // A point is never voted wood without being labelled wood.
        assert!(mask.iter().zip(&votes).all(|(&m, &v)| m || v == 0.0));
    }

    #[test]
    fn weights_are_checked() {
        let pts = vec![[0.0, 0.0, 0.0]; 4];
        let p = QsmParams::default();
        assert!(build_qsm_weighted(&pts, None, &p, &[1.0; 3]).is_err());
        assert!(build_qsm_weighted(&pts, None, &p, &[1.0, f64::NAN, 1.0, 1.0]).is_err());
        assert!(build_qsm_weighted(&pts, None, &p, &[1.0, 1.5, 1.0, 1.0]).is_err());
        assert!(build_qsm_weighted(&pts, None, &p, &[1.0, -0.1, 1.0, 1.0]).is_err());
        assert!(build_qsm_weighted(&pts, None, &QsmParams { min_weight: 0.5, ..Default::default() }, &[0.1; 4]).is_err());
    }
}
