// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Marker-free coregistration: global matching of stem maps.
//!
//! Each scan is reduced to its stems (position at breast height, diameter,
//! quality). The distance between two stems does not depend on the unknown
//! transform, so a pair of stems in one scan can only correspond to a pair at
//! the same separation in the other. Sorting the target's pair table by
//! separation turns the search into a binary search, and two correspondences
//! fix a 4-DoF transform (yaw and translation), the right model for
//! gravity-levelled TLS; ICP recovers any residual tilt afterwards.

use crate::transform::Transform;
use crate::Point;
use nalgebra::{Matrix3, Vector3};

/// Tunables of [`match_stem_maps`].
#[derive(Debug, Clone)]
pub struct MatchParams {
    /// Stem pairs closer than this are ambiguous and ignored (m).
    pub min_pair_distance: f64,
    /// Stem pairs further apart are rarely seen from both scans (m).
    pub max_pair_distance: f64,
    /// Tolerance on pair separation (m).
    pub pair_distance_tolerance: f64,
    /// Horizontal stem-to-stem distance that counts as a match (m).
    pub inlier_tolerance: f64,
    pub diameter_rel_tolerance: f64,
    pub diameter_abs_tolerance: f64,
    pub use_diameters: bool,
    /// Only the best stems of each scan take part (cost is quadratic).
    pub max_stems: usize,
    pub max_hypotheses: usize,
    pub min_inliers: usize,
    /// Stop once a hypothesis has this many inliers.
    pub early_exit_inliers: usize,
    /// A rival hypothesis must differ by this much translation (m) ...
    pub distinct_translation: f64,
    /// ... or this much yaw (degrees) to count as a different solution.
    pub distinct_yaw_deg: f64,
    pub refine_iterations: usize,
}

impl Default for MatchParams {
    fn default() -> Self {
        MatchParams {
            min_pair_distance: 2.0,
            max_pair_distance: 35.0,
            pair_distance_tolerance: 0.25,
            inlier_tolerance: 0.40,
            diameter_rel_tolerance: 0.30,
            diameter_abs_tolerance: 0.04,
            use_diameters: true,
            max_stems: 70,
            max_hypotheses: 60_000,
            min_inliers: 4,
            early_exit_inliers: 40,
            distinct_translation: 1.0,
            distinct_yaw_deg: 5.0,
            refine_iterations: 6,
        }
    }
}

/// One scan's stems.
#[derive(Debug, Clone, Default)]
pub struct StemMap {
    /// Axis position at breast height, as an absolute elevation.
    pub positions: Vec<Point>,
    pub diameters: Vec<f64>,
    pub qualities: Vec<f64>,
}

impl StemMap {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Indices of the `n` highest-quality stems, best first (ties keep input order).
    fn top(&self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len()).collect();
        idx.sort_by(|&a, &b| self.qualities[b].partial_cmp(&self.qualities[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(n);
        idx
    }
}

/// Outcome of matching two stem maps; `transform` maps source into target.
#[derive(Debug, Clone)]
pub struct StemMatch {
    pub transform: Transform,
    pub n_inliers: usize,
    pub inlier_rmse: f64,
    pub score: f64,
    /// `(source index, target index)` into the stem maps passed in.
    pub correspondences: Vec<(usize, usize)>,
    pub n_source: usize,
    pub n_target: usize,
    pub success: bool,
    /// Inliers of the best distinctly different hypothesis over the best's:
    /// near 1 means the pattern matches itself elsewhere (a planted lattice).
    pub ambiguity: f64,
    /// That rival, refined, so ICP can decide between the two.
    pub rival: Option<Box<StemMatch>>,
}

impl StemMatch {
    fn empty(n_source: usize, n_target: usize) -> Self {
        StemMatch {
            transform: Transform::identity(),
            n_inliers: 0,
            inlier_rmse: f64::INFINITY,
            score: 0.0,
            correspondences: Vec::new(),
            n_source,
            n_target,
            success: false,
            ambiguity: 0.0,
            rival: None,
        }
    }

    fn yaw(&self) -> f64 {
        self.transform.0[(1, 0)].atan2(self.transform.0[(0, 0)])
    }
}

/// Pairwise summation in numpy's order (`np.add.reduce` on a contiguous
/// float64 array: eight accumulators up to 128 values, recursive halves
/// beyond). The order is fixed so that sums, and the decisions taken on them,
/// are deterministic to the bit.
pub fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
    }
}

/// Least-squares yaw + 3-D translation between paired points.
///
/// Computed in a fixed arithmetic order (uniform weights `1/n`, numpy's
/// summation orders, the rotation from `so3_exp([0, 0, yaw])` and `R @ mu_s`
/// as OpenBLAS's fused dgemv gives it), so that the hypothesis search takes
/// the same decisions on every run.
pub fn kabsch_yaw(src: &[Point], dst: &[Point]) -> Transform {
    let n = src.len().min(dst.len());
    let w = 1.0 / n.max(1) as f64;
    let mut ms = [0.0; 3];
    let mut md = [0.0; 3];
    for (i, (a, b)) in src.iter().zip(dst).enumerate() {
        for k in 0..3 {
            if i == 0 {
                ms[k] = w * a[k];
                md[k] = w * b[k];
            } else {
                ms[k] += w * a[k];
                md[k] += w * b[k];
            }
        }
    }
    let mut num = Vec::with_capacity(n);
    let mut den = Vec::with_capacity(n);
    for (a, b) in src.iter().zip(dst) {
        let (ax, ay) = (a[0] - ms[0], a[1] - ms[1]);
        let (bx, by) = (b[0] - md[0], b[1] - md[1]);
        num.push(w * (ax * by - ay * bx));
        den.push(w * (ax * bx + ay * by));
    }
    let yaw = numpy_sum(&num).atan2(numpy_sum(&den));
    // so3_exp([0, 0, yaw]) = I + sin(t)/t K + (1 - cos t)/t^2 K^2, t = |yaw|.
    let th = (yaw * yaw).sqrt();
    let (r00, r01, r10) = if th < 1e-8 {
        (1.0 + 0.5 * -(yaw * yaw), -yaw, yaw)
    } else {
        let (s1, c1) = (th.sin() / th, (1.0 - th.cos()) / (th * th));
        (1.0 + c1 * -(yaw * yaw), s1 * -yaw, s1 * yaw)
    };
    let r = Matrix3::new(r00, r01, 0.0, r10, r00, 0.0, 0.0, 0.0, 1.0);
    let t = Vector3::new(md[0] - r00.mul_add(ms[0], r01 * ms[1]), md[1] - r10.mul_add(ms[0], r00 * ms[1]), md[2] - ms[2]);
    Transform::from_rt(r, t)
}

struct Maps<'a> {
    src: Vec<Point>,
    dst: Vec<Point>,
    src_d: Vec<f64>,
    dst_d: Vec<f64>,
    src_q: Vec<f64>,
    src_id: Vec<usize>,
    dst_id: Vec<usize>,
    p: &'a MatchParams,
}

impl Maps<'_> {
    fn diameter_ok(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.p.diameter_abs_tolerance.max(self.p.diameter_rel_tolerance * a.max(b))
    }

    /// Count one-to-one stem correspondences under `t` (indices local to the top-N maps).
    fn score(&self, t: &Transform) -> StemMatch {
        let (ns, nd) = (self.src.len(), self.dst.len());
        let tol2 = self.p.inlier_tolerance * self.p.inlier_tolerance;
        // Nearest target stem for each source stem, within tolerance.
        let mut hits: Vec<(usize, usize, f64)> = Vec::new();
        for (i, q) in self.src.iter().enumerate() {
            let m = t.apply(q);
            let mut best = (usize::MAX, f64::INFINITY);
            for (j, r) in self.dst.iter().enumerate() {
                let d2 = (m[0] - r[0]).powi(2) + (m[1] - r[1]).powi(2);
                if d2 < best.1 {
                    best = (j, d2);
                }
            }
            // cKDTree's distance_upper_bound is exclusive.
            if best.1 < tol2 && (!self.p.use_diameters || self.diameter_ok(self.src_d[i], self.dst_d[best.0])) {
                hits.push((i, best.0, best.1.sqrt()));
            }
        }
        // One-to-one: of the source stems claiming a target stem, keep the closest.
        hits.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.partial_cmp(&b.2).unwrap()));
        hits.dedup_by(|b, a| a.1 == b.1);
        let n = hits.len();
        let mut out = StemMatch::empty(ns, nd);
        out.transform = *t;
        if n == 0 {
            return out;
        }
        let sq: Vec<f64> = hits.iter().map(|h| h.2 * h.2).collect();
        let rmse = (numpy_sum(&sq) / n as f64).sqrt();
        out.n_inliers = n;
        out.inlier_rmse = rmse;
        // An extra inlier is worth more than a marginally tighter fit.
        out.score = n as f64 * (1.0 - (rmse / self.p.inlier_tolerance).min(1.0) * 0.5);
        out.correspondences = hits.iter().map(|h| (h.0, h.1)).collect();
        out
    }

    /// Alternate re-estimation and correspondence search.
    fn refine(&self, start: StemMatch) -> StemMatch {
        let mut best = start;
        for _ in 0..self.p.refine_iterations {
            if best.correspondences.len() < 2 {
                break;
            }
            let a: Vec<Point> = best.correspondences.iter().map(|&(i, _)| self.src[i]).collect();
            let b: Vec<Point> = best.correspondences.iter().map(|&(_, j)| self.dst[j]).collect();
            let cand = self.score(&kabsch_yaw(&a, &b));
            if cand.n_inliers > best.n_inliers || (cand.n_inliers == best.n_inliers && cand.inlier_rmse < best.inlier_rmse) {
                best = cand;
            } else {
                break;
            }
        }
        best
    }

    fn distinct(&self, a: &StemMatch, b: &StemMatch) -> bool {
        if a.n_inliers == 0 || b.n_inliers == 0 {
            return false;
        }
        let d = a.transform.0.fixed_view::<3, 1>(0, 3) - b.transform.0.fixed_view::<3, 1>(0, 3);
        let mut dyaw = (a.yaw() - b.yaw()).abs().to_degrees();
        dyaw = dyaw.min(360.0 - dyaw);
        d.norm() > self.p.distinct_translation || dyaw > self.p.distinct_yaw_deg
    }

    /// Map local top-N indices back to the caller's indices.
    fn to_caller(&self, mut m: StemMatch, n_source: usize, n_target: usize) -> StemMatch {
        m.correspondences = m.correspondences.iter().map(|&(i, j)| (self.src_id[i], self.dst_id[j])).collect();
        m.n_source = n_source;
        m.n_target = n_target;
        m
    }
}

fn pair_table(xy: &[Point], lo: f64, hi: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..xy.len() {
        for j in i + 1..xy.len() {
            // As np.linalg.norm(axis=1) computes it (not hypot).
            let (dx, dy) = (xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]);
            let d = (dx * dx + dy * dy).sqrt();
            if d >= lo && d <= hi {
                out.push((i, j, d));
            }
        }
    }
    out
}

/// Find the yaw + translation that aligns `source` stems onto `target` stems,
/// with no initial guess. Check `success` before using `transform`.
pub fn match_stem_maps(source: &StemMap, target: &StemMap, p: &MatchParams) -> StemMatch {
    let (src_id, dst_id) = (source.top(p.max_stems), target.top(p.max_stems));
    let m = Maps {
        src: src_id.iter().map(|&i| source.positions[i]).collect(),
        dst: dst_id.iter().map(|&i| target.positions[i]).collect(),
        src_d: src_id.iter().map(|&i| source.diameters[i]).collect(),
        dst_d: dst_id.iter().map(|&i| target.diameters[i]).collect(),
        src_q: src_id.iter().map(|&i| source.qualities[i].max(1e-3)).collect(),
        src_id,
        dst_id,
        p,
    };
    // Reported as the sizes of the top-N maps that were matched.
    let (ns, nd) = (m.src.len(), m.dst.len());
    let empty = StemMatch::empty(ns, nd);
    if m.src.len() < 2 || m.dst.len() < 2 {
        return empty;
    }
    let mut dst_pairs = pair_table(&m.dst, p.min_pair_distance, p.max_pair_distance);
    let mut src_pairs = pair_table(&m.src, p.min_pair_distance, p.max_pair_distance);
    if dst_pairs.is_empty() || src_pairs.is_empty() {
        return empty;
    }
    dst_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let dst_d: Vec<f64> = dst_pairs.iter().map(|x| x.2).collect();
    // Pairs of the most reliable stems first: descending product of the two
    // qualities (floored at 1e-3), ties in (i, j) enumeration order.
    src_pairs.sort_by(|a, b| (m.src_q[b.0] * m.src_q[b.1]).partial_cmp(&(m.src_q[a.0] * m.src_q[a.1])).unwrap_or(std::cmp::Ordering::Equal));

    let mut best = StemMatch::empty(ns, nd);
    let mut runner = StemMatch::empty(ns, nd);
    let mut hypotheses = 0;
    let tol = p.pair_distance_tolerance;
    'outer: for &(a1, a2, da) in &src_pairs {
        if hypotheses >= p.max_hypotheses {
            break;
        }
        let lo = dst_d.partition_point(|&d| d < da - tol);
        let hi = dst_d.partition_point(|&d| d <= da + tol);
        for &(b1, b2, _) in &dst_pairs[lo..hi] {
            for (u, v) in [(b1, b2), (b2, b1)] {
                if p.use_diameters && !(m.diameter_ok(m.src_d[a1], m.dst_d[u]) && m.diameter_ok(m.src_d[a2], m.dst_d[v])) {
                    continue;
                }
                hypotheses += 1;
                let t = kabsch_yaw(&[m.src[a1], m.src[a2]], &[m.dst[u], m.dst[v]]);
                let r = m.score(&t);
                if r.score > best.score {
                    if m.distinct(&r, &best) {
                        runner = std::mem::replace(&mut best, r);
                    } else {
                        best = r;
                    }
                } else if r.score > runner.score && m.distinct(&r, &best) {
                    runner = r;
                }
            }
            if best.n_inliers >= p.early_exit_inliers {
                break 'outer;
            }
        }
    }
    if best.n_inliers < p.min_inliers {
        return empty;
    }
    let ambiguity = if best.n_inliers > 0 { (runner.n_inliers as f64 / best.n_inliers as f64).min(1.0) } else { 0.0 };
    let mut out = m.refine(best);
    out.success = out.n_inliers >= p.min_inliers;
    out.ambiguity = ambiguity;
    if runner.n_inliers >= p.min_inliers {
        let mut rival = m.refine(runner);
        rival.success = rival.n_inliers >= p.min_inliers;
        out.rival = Some(Box::new(m.to_caller(rival, ns, nd)));
    }
    m.to_caller(out, ns, nd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn recovers_a_yaw_and_shift_between_overlapping_maps() {
        let mut seed = 3;
        let world: Vec<(Point, f64)> = (0..60).map(|_| ([lcg(&mut seed) * 40.0, lcg(&mut seed) * 40.0, 1.3 + lcg(&mut seed)], 0.1 + 0.5 * lcg(&mut seed))).collect();
        // Target sees the west part, source the east part, in a frame rotated 37 deg and shifted.
        let truth = Transform::from_rt(nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), 37f64.to_radians()).into_inner(), Vector3::new(5.0, -3.0, 0.8));
        let inv = truth.inverse().unwrap();
        let mut src = StemMap::default();
        let mut dst = StemMap::default();
        for (k, (q, d)) in world.iter().enumerate() {
            if q[0] < 28.0 {
                dst.positions.push(*q);
                dst.diameters.push(*d);
                dst.qualities.push(1.0);
            }
            if q[0] > 12.0 {
                let jitter = [0.02 * (lcg(&mut seed) - 0.5), 0.02 * (lcg(&mut seed) - 0.5), 0.0];
                let p = inv.apply(&[q[0] + jitter[0], q[1] + jitter[1], q[2]]);
                src.positions.push(p);
                src.diameters.push(*d);
                src.qualities.push(1.0 - k as f64 / 100.0);
            }
        }
        let r = match_stem_maps(&src, &dst, &MatchParams::default());
        assert!(r.success && r.n_inliers >= 10, "{} inliers", r.n_inliers);
        let err = (r.transform.0 - truth.0).abs().max();
        assert!(err < 0.05, "transform off by {err}");
        for &(i, j) in &r.correspondences {
            let moved = r.transform.apply(&src.positions[i]);
            assert!((moved[0] - dst.positions[j][0]).hypot(moved[1] - dst.positions[j][1]) < 0.1);
        }
    }
}
