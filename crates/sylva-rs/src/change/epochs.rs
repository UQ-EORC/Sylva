// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Alignment of a second epoch onto a first on features that do not change.
//!
//! Between surveys the canopy grows, loses limbs and moves in the wind, so a
//! registration over all points is pulled by the very change it should
//! reveal. Stems (their axes at breast height) and the terrain are the
//! stable parts of a plot: a stem thickens but its axis stays where it
//! was, and the ground changes little. Starting from a coarse transform
//! (from [`crate::coreg::pipeline::register_pair`]), the alignment is
//! refined by robust Gauss-Newton on two kinds of residual: the horizontal
//! offset of each matched stem (x, y, rotation about z) and the height of
//! the second epoch's terrain above the first's (z, and the tilts through
//! the terrain's extent). Each kind is weighted by its own robust spread
//! (Huber 1964 weights), and the covariance of the six parameters follows
//! from the normal equations, so the uncertainty of the alignment at any
//! point of the plot is known.

use nalgebra::{Matrix3, Matrix6, Rotation3, Vector3, Vector6};

use crate::coreg::ground::GroundModel;
use crate::error::{Error, Result};
use crate::{Point, Transform};
use super::{positive, non_negative};

/// Settings of [`align_on_stable`].
#[derive(Debug, Clone)]
pub struct AlignParams {
    /// Use stems (x, y and rotation about z).
    pub use_stems: bool,
    /// Use the terrain (z and the tilts).
    pub use_ground: bool,
    /// Farthest a stem of the second epoch may lie from its match (m)
    /// under the current transform.
    pub stem_tolerance: f64,
    /// Largest relative DBH difference of a stem pair.
    pub dbh_tolerance: f64,
    /// Spacing (m) of the terrain samples.
    pub ground_spacing: f64,
    /// Terrain errors are taken as correlated within blocks of this side
    /// (m): the terrain samples together weigh as much as one independent
    /// sample per block, so a terrain model's smooth, spatially coherent
    /// errors do not pass for precision.
    pub ground_block: f64,
    /// Smallest error assumed for one stem position (per axis) and one
    /// terrain sample (m): no bark surface or terrain model is known better,
    /// whatever the residuals of an unusually clean plot suggest.
    pub stem_floor: f64,
    pub ground_floor: f64,
    /// Residuals beyond this many robust standard deviations are
    /// down-weighted (Huber).
    pub huber: f64,
    pub iterations: usize,
}

impl Default for AlignParams {
    fn default() -> Self {
        AlignParams { use_stems: true, use_ground: true, stem_tolerance: 0.3, dbh_tolerance: 0.35, ground_spacing: 1.0, ground_block: 5.0, stem_floor: 0.002, ground_floor: 0.005, huber: 2.0, iterations: 30 }
    }
}

/// Result of [`align_on_stable`].
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Maps the second epoch onto the first.
    pub transform: Transform,
    /// Covariance of `(rx, ry, rz, tx, ty, tz)`: a small rotation (rad)
    /// about `centre` followed by a translation (m). Rows of a parameter
    /// the features do not constrain are NaN.
    pub covariance: Matrix6<f64>,
    pub centre: Point,
    /// One-sigma uncertainty (m) of x, y, z at `centre`.
    pub sigma_xyz: [f64; 3],
    /// RMS over the evaluation points of the one-sigma 3-D displacement
    /// uncertainty (m), and of its horizontal and vertical parts.
    pub registration_sigma: f64,
    pub sigma_horizontal: f64,
    pub sigma_vertical: f64,
    /// Matched stems `(reference, new)` and their horizontal residuals.
    pub stem_pairs: Vec<(usize, usize)>,
    pub stem_residuals: Vec<[f64; 2]>,
    /// Terrain samples used and their vertical residuals.
    pub n_ground: usize,
    pub ground_residuals: Vec<f64>,
    /// Robust spread of one stem residual component and one terrain
    /// residual, at least the floors.
    pub stem_sigma: f64,
    pub ground_sigma: f64,
    pub iterations: usize,
}

fn mad_sigma(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let med = v[v.len() / 2];
    let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    1.4826 * dev[dev.len() / 2]
}

/// Terrain samples of a ground model: the observed cells, thinned to about
/// `spacing` m, as `(x, y, z)`.
pub fn ground_samples(g: &GroundModel, spacing: f64) -> Vec<Point> {
    let step = ((spacing / g.cell_size).round() as usize).max(1);
    let mut out = Vec::new();
    for r in (0..g.ny).step_by(step) {
        for c in (0..g.nx).step_by(step) {
            let k = r * g.nx + c;
            if g.observed[k] && g.elevation[k].is_finite() {
                out.push([g.origin[0] + c as f64 * g.cell_size, g.origin[1] + r as f64 * g.cell_size, g.elevation[k]]);
            }
        }
    }
    out
}

/// Jacobian rows of a point displacement `omega x v + t` (v from the centre).
fn rows(v: &Point) -> [[f64; 6]; 3] {
    [[0.0, v[2], -v[1], 1.0, 0.0, 0.0], [-v[2], 0.0, v[0], 0.0, 1.0, 0.0], [v[1], -v[0], 0.0, 0.0, 0.0, 1.0]]
}

/// Refine `initial` (new onto reference) on stems and terrain; see the
/// module documentation. Stems are `(x, y, z, dbh)` at breast height;
/// `ground_new` are terrain samples of the second epoch in its own frame,
/// compared with the reference terrain model. `evaluate` are the points at
/// which the uncertainty is summarised (the reference stems if empty).
pub fn align_on_stable(ref_stems: &[[f64; 4]], new_stems: &[[f64; 4]], ref_ground: Option<&GroundModel>, ground_new: &[Point], initial: &Transform, evaluate: &[Point], p: &AlignParams) -> Result<Alignment> {
    if !p.use_stems && !p.use_ground {
        return Err(Error::invalid("use at least one of stems and ground"));
    }
    if !positive(p.stem_tolerance) || !positive(p.huber) || !positive(p.ground_block) || !non_negative(p.stem_floor) || !non_negative(p.ground_floor) {
        return Err(Error::invalid("stem_tolerance, ground_block and huber must be positive and the floors >= 0"));
    }
    let use_ground = p.use_ground && ref_ground.is_some() && !ground_new.is_empty();
    let use_stems = p.use_stems && !ref_stems.is_empty() && !new_stems.is_empty();
    if !use_ground && !use_stems {
        return Err(Error::invalid("no stable features: the stems or the terrain of one epoch are missing"));
    }
    // Parameters the features constrain.
    let free: [bool; 6] = [use_ground, use_ground, use_stems, use_stems, use_stems, use_ground];
    let centre = {
        let src: Vec<Point> = if use_stems { ref_stems.iter().map(|s| [s[0], s[1], s[2]]).collect() } else { ground_new.iter().map(|q| initial.apply(q)).collect() };
        let n = src.len() as f64;
        [src.iter().map(|q| q[0]).sum::<f64>() / n, src.iter().map(|q| q[1]).sum::<f64>() / n, src.iter().map(|q| q[2]).sum::<f64>() / n]
    };
    let mut t = *initial;
    let mut info = Matrix6::<f64>::zeros();
    let (mut stem_pairs, mut stem_res, mut ground_res) = (Vec::new(), Vec::new(), Vec::new());
    let (mut s_sigma, mut g_sigma) = (f64::NAN, f64::NAN);
    let mut n_ground = 0usize;
    let mut iterations = 0usize;
    for it in 0..p.iterations.max(1) {
        iterations = it + 1;
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        // Stems: mutual nearest neighbours within the tolerance.
        stem_pairs.clear();
        stem_res.clear();
        let mut stem_v: Vec<Point> = Vec::new();
        if use_stems {
            let moved: Vec<Point> = new_stems.iter().map(|s| t.apply(&[s[0], s[1], s[2]])).collect();
            let nearest = |q: &Point, set: &[Point]| -> Option<(usize, f64)> {
                set.iter().enumerate().map(|(k, s)| (k, (q[0] - s[0]).hypot(q[1] - s[1]))).min_by(|a, b| a.1.total_cmp(&b.1))
            };
            let refs: Vec<Point> = ref_stems.iter().map(|s| [s[0], s[1], s[2]]).collect();
            for (j, q) in moved.iter().enumerate() {
                let Some((i, d)) = nearest(q, &refs) else { continue };
                if d > p.stem_tolerance || nearest(&refs[i], &moved).map(|m| m.0) != Some(j) {
                    continue;
                }
                let (da, db) = (ref_stems[i][3], new_stems[j][3]);
                if da.is_finite() && db.is_finite() && da.max(db) > 0.0 && (db - da).abs() / da.max(db) > p.dbh_tolerance {
                    continue;
                }
                stem_pairs.push((i, j));
                stem_res.push([q[0] - refs[i][0], q[1] - refs[i][1]]);
                stem_v.push([q[0] - centre[0], q[1] - centre[1], q[2] - centre[2]]);
            }
            let mut comps: Vec<f64> = stem_res.iter().flat_map(|r| [r[0], r[1]]).collect();
            s_sigma = mad_sigma(&mut comps).max(p.stem_floor).max(1e-6);
        }
        // Terrain: height of the new terrain above the reference one.
        ground_res.clear();
        let mut ground_rows: Vec<[f64; 6]> = Vec::new();
        let mut ground_used: Vec<Point> = Vec::new();
        if use_ground {
            let g = ref_ground.expect("checked");
            let h = 0.5 * g.cell_size;
            for q in ground_new {
                let m = t.apply(q);
                if !g.support(&[[m[0], m[1]]])[0] {
                    continue;
                }
                let z0 = g.height_at_point(m[0], m[1]);
                let gx = (g.height_at_point(m[0] + h, m[1]) - g.height_at_point(m[0] - h, m[1])) / (2.0 * h);
                let gy = (g.height_at_point(m[0], m[1] + h) - g.height_at_point(m[0], m[1] - h)) / (2.0 * h);
                let v = [m[0] - centre[0], m[1] - centre[1], m[2] - centre[2]];
                let j = rows(&v);
                ground_res.push(m[2] - z0);
                ground_used.push(m);
                ground_rows.push(std::array::from_fn(|k| j[2][k] - gx * j[0][k] - gy * j[1][k]));
            }
            let mut r = ground_res.clone();
            g_sigma = mad_sigma(&mut r).max(p.ground_floor).max(1e-6);
            n_ground = ground_res.len();
        }
        let mut add = |row: &[f64; 6], r: f64, w: f64| {
            let jv = Vector6::from_row_slice(row);
            jtj += w * jv * jv.transpose();
            jtr += w * r * jv;
        };
        for (res, v) in stem_res.iter().zip(&stem_v) {
            let n = res[0].hypot(res[1]);
            let w = (p.huber * s_sigma / n.max(1e-12)).min(1.0) / (s_sigma * s_sigma);
            let j = rows(v);
            add(&j[0], res[0], w);
            add(&j[1], res[1], w);
        }
        let g_med = {
            let mut r = ground_res.clone();
            r.sort_by(f64::total_cmp);
            r.get(r.len() / 2).copied().unwrap_or(0.0)
        };
        let blocks: std::collections::HashSet<(i64, i64)> = ground_used.iter().map(|q| ((q[0] / p.ground_block).floor() as i64, (q[1] / p.ground_block).floor() as i64)).collect();
        let share = if ground_res.is_empty() { 1.0 } else { (blocks.len() as f64 / ground_res.len() as f64).min(1.0) };
        for (r, row) in ground_res.iter().zip(&ground_rows) {
            let w = share * (p.huber * g_sigma / (r - g_med).abs().max(1e-12)).min(1.0) / (g_sigma * g_sigma);
            add(row, *r, w);
        }
        // Solve for the free parameters only.
        let idx: Vec<usize> = (0..6).filter(|&k| free[k]).collect();
        let n = idx.len();
        let a = nalgebra::DMatrix::from_fn(n, n, |r, c| jtj[(idx[r], idx[c])]);
        let b = nalgebra::DVector::from_fn(n, |r, _| -jtr[idx[r]]);
        let Some(x) = a.clone().cholesky().map(|c| c.solve(&b)) else {
            return Err(Error::invalid(format!("the stable features do not constrain the alignment ({} stem pairs, {} terrain samples)", stem_pairs.len(), n_ground)));
        };
        let mut delta = [0.0; 6];
        for (r, &k) in idx.iter().enumerate() {
            delta[k] = x[r];
        }
        let rot = Rotation3::new(Vector3::new(delta[0], delta[1], delta[2])).into_inner();
        let c = Vector3::new(centre[0], centre[1], centre[2]);
        let step = Transform::from_rt(rot, c + Vector3::new(delta[3], delta[4], delta[5]) - rot * c);
        t = step.compose(&t);
        info = jtj;
        let moved = (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]).sqrt() + 30.0 * (delta[0].abs() + delta[1].abs() + delta[2].abs());
        if moved < 1e-7 {
            break;
        }
    }
    if use_stems && stem_pairs.len() < 3 && !use_ground {
        return Err(Error::invalid(format!("only {} stems matched; the stems alone cannot align the epochs", stem_pairs.len())));
    }
    // Covariance of the free parameters; NaN rows for the others.
    let idx: Vec<usize> = (0..6).filter(|&k| free[k]).collect();
    let n = idx.len();
    let a = nalgebra::DMatrix::from_fn(n, n, |r, c| info[(idx[r], idx[c])]);
    let inv = a.try_inverse().ok_or_else(|| Error::invalid("singular normal equations"))?;
    let mut cov = Matrix6::<f64>::from_element(f64::NAN);
    for (r, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            cov[(i, j)] = inv[(r, c)];
        }
    }
    let point_cov = |q: &Point| -> Matrix3<f64> {
        let v = [q[0] - centre[0], q[1] - centre[1], q[2] - centre[2]];
        let j = rows(&v);
        Matrix3::from_fn(|r, c| {
            let mut s = 0.0;
            for a in 0..6 {
                for b in 0..6 {
                    if free[a] && free[b] {
                        s += j[r][a] * cov[(a, b)] * j[c][b];
                    }
                }
            }
            s
        })
    };
    let pc = point_cov(&centre);
    let unconstrained = |k: usize| -> bool { !free[k] && !free[k + 3] };
    let sig = |k: usize, v: f64| if free[k + 3] { v.max(0.0).sqrt() } else { f64::NAN };
    let sigma_xyz = [sig(0, pc[(0, 0)]), sig(1, pc[(1, 1)]), sig(2, pc[(2, 2)])];
    let evals: Vec<Point> = if evaluate.is_empty() { if use_stems { ref_stems.iter().map(|s| [s[0], s[1], s[2]]).collect() } else { vec![centre] } } else { evaluate.to_vec() };
    let (mut sh, mut sv) = (0.0, 0.0);
    for q in &evals {
        let m = point_cov(q);
        sh += m[(0, 0)] + m[(1, 1)];
        sv += m[(2, 2)];
    }
    let ne = evals.len() as f64;
    let (sh, sv) = ((sh / ne).sqrt(), (sv / ne).sqrt());
    let sh = if unconstrained(0) || unconstrained(1) || !free[3] { f64::NAN } else { sh };
    let sv = if !free[5] { f64::NAN } else { sv };
    Ok(Alignment {
        transform: t,
        covariance: cov,
        centre,
        sigma_xyz,
        registration_sigma: (sh * sh + sv * sv).sqrt(),
        sigma_horizontal: sh,
        sigma_vertical: sv,
        stem_pairs,
        stem_residuals: stem_res,
        n_ground,
        ground_residuals: ground_res,
        stem_sigma: s_sigma,
        ground_sigma: g_sigma,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terrain(x: f64, y: f64) -> f64 {
        0.05 * x + 0.2 * (y / 3.0).sin()
    }

    fn model(shift: &Transform) -> (GroundModel, Vec<Point>) {
        // Reference terrain on a 0.5 m grid; the new terrain is the same
        // surface seen through `shift`.
        let (nx, ny, cs) = (61, 61, 0.5);
        let elevation: Vec<f64> = (0..ny).flat_map(|r| (0..nx).map(move |c| terrain(c as f64 * cs, r as f64 * cs))).collect();
        let g = GroundModel { nx, ny, elevation, origin: [0.0, 0.0], cell_size: cs, observed: vec![true; nx * ny] };
        let mut samples = Vec::new();
        for r in (4..ny - 4).step_by(2) {
            for c in (4..nx - 4).step_by(2) {
                let (x, y) = (c as f64 * cs, r as f64 * cs);
                samples.push(shift.apply(&[x, y, terrain(x, y)]));
            }
        }
        (g, samples)
    }

    #[test]
    fn recovers_a_known_offset() {
        let offset = Transform::translation(0.05, -0.03, 0.02).compose(&Transform::from_roll_pitch_yaw(0.05, -0.04, 0.3));
        let (g, samples) = model(&offset);
        let refs: Vec<[f64; 4]> = [(5.0, 5.0), (20.0, 6.0), (8.0, 22.0), (25.0, 25.0), (15.0, 15.0)].iter().map(|&(x, y)| [x, y, terrain(x, y) + 1.3, 0.3]).collect();
        let news: Vec<[f64; 4]> = refs.iter().map(|s| {
            let q = offset.apply(&[s[0], s[1], s[2]]);
            [q[0], q[1], q[2], 0.31]
        }).collect();
        let a = align_on_stable(&refs, &news, Some(&g), &samples, &Transform::identity(), &[], &AlignParams::default()).unwrap();
        let back = a.transform.compose(&offset);
        assert!((back.0 - nalgebra::Matrix4::identity()).abs().max() < 1e-6, "{}", back.0);
        assert_eq!(a.stem_pairs.len(), 5);
        assert!(a.sigma_xyz.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn stems_only_leave_z_unconstrained() {
        let refs: Vec<[f64; 4]> = [(5.0, 5.0), (20.0, 6.0), (8.0, 22.0), (25.0, 25.0)].iter().map(|&(x, y)| [x, y, 1.3, 0.3]).collect();
        let shift = Transform::translation(0.1, 0.0, 0.0);
        let news: Vec<[f64; 4]> = refs.iter().map(|s| [s[0] + 0.1, s[1], s[2], s[3]]).collect();
        let p = AlignParams { use_ground: false, ..Default::default() };
        let a = align_on_stable(&refs, &news, None, &[], &Transform::identity(), &[], &p).unwrap();
        let back = a.transform.compose(&shift);
        assert!((back.0 - nalgebra::Matrix4::identity()).abs().max() < 1e-9);
        assert!(a.sigma_xyz[2].is_nan() && a.sigma_vertical.is_nan());
        assert!(align_on_stable(&[], &[], None, &[], &Transform::identity(), &[], &AlignParams::default()).is_err());
    }
}
