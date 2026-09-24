// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Portions adapted from rayvoxel (raycloudtools fork, Josh Rivory), Copyright (c)
// 2020 CSIRO, under the CSIRO licence in THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Pulse traversal: the accumulation half of rayvoxel (`VoxelProcessor`).

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicU8, Ordering::Relaxed};

use rayon::prelude::*;

use super::{foliage, unbounded_end, Attenuation, RayVoxels, VoxelInputs, VoxelParams, WeightMethod, F, I};
use crate::transform::{add, norm, scale, sub};
use crate::Point;

/// Grid placement.
#[derive(Debug, Clone)]
pub(crate) struct Geom {
    pub origin: Point,
    pub size: f64,
    pub shape: [usize; 3],
    max: Point,
}

impl Geom {
    pub fn new(origin: Point, size: f64, shape: [usize; 3]) -> Self {
        let max = [origin[0] + shape[0] as f64 * size, origin[1] + shape[1] as f64 * size, origin[2] + shape[2] as f64 * size];
        Geom { origin, size, shape, max }
    }

    #[inline]
    pub fn flat(&self, p: [i64; 3]) -> Option<usize> {
        if (0..3).all(|k| p[k] >= 0 && (p[k] as usize) < self.shape[k]) {
            Some(p[0] as usize + self.shape[0] * (p[1] as usize + self.shape[1] * p[2] as usize))
        } else {
            None
        }
    }

    #[inline]
    pub fn to_vox(&self, p: &Point) -> Point {
        [(p[0] - self.origin[0]) / self.size, (p[1] - self.origin[1]) / self.size, (p[2] - self.origin[2]) / self.size]
    }

    #[inline]
    pub fn cell_of(&self, p: &Point) -> [i64; 3] {
        let v = self.to_vox(p);
        [v[0].floor() as i64, v[1].floor() as i64, v[2].floor() as i64]
    }

    fn center(&self, p: [i64; 3]) -> Point {
        [
            self.origin[0] + (p[0] as f64 + 0.5) * self.size,
            self.origin[1] + (p[1] as f64 + 0.5) * self.size,
            self.origin[2] + (p[2] as f64 + 0.5) * self.size,
        ]
    }

    /// Clip a segment to the grid box shrunk by `eps` (raylib `Cuboid::clipRay`).
    fn clip(&self, start: &mut Point, end: &mut Point, eps: f64) -> bool {
        let mut near = 0.0f64;
        let mut far = 1.0f64;
        let dir = sub(end, start);
        for ax in 0..3 {
            let centre = 0.5 * (self.origin[ax] + self.max[ax]);
            let extent = 0.5 * (self.max[ax] - self.origin[ax]) - eps;
            let to_centre = centre - start[ax];
            let s = if dir[ax] > 0.0 { 1.0 } else { -1.0 };
            let (n, f) = (to_centre - s * extent, to_centre + s * extent);
            if dir[ax] != 0.0 {
                near = near.max(n / dir[ax]);
                far = far.min(f / dir[ax]);
            } else if (n > 0.0) == (f > 0.0) {
                return false;
            }
        }
        if far <= near {
            return false;
        }
        let s = add(start, &scale(&dir, near));
        let e = sub(end, &scale(&dir, 1.0 - far));
        *start = s;
        *end = e;
        true
    }
}

/// Amanatides & Woo (1987) walk in voxel units (raylib `walkGrid`). `visit(cell,
/// in_length, out_length, max_length)` returns `true` to stop.
pub(crate) fn walk_grid(start: &Point, end: &Point, mut visit: impl FnMut([i64; 3], f64, f64, f64) -> bool) {
    let mut dir = sub(end, start);
    let mut max_length = norm(&dir);
    if !(max_length > 0.0) {
        return;
    }
    let mut p = [start[0].floor() as i64, start[1].floor() as i64, start[2].floor() as i64];
    let step = [sign(dir[0]), sign(dir[1]), sign(dir[2])];
    for d in &mut dir {
        *d /= max_length;
    }
    // Stay out of the neighbouring voxel when the end point sits on a face.
    max_length -= 1e-10f32 as f64;
    let mut lengths = [0.0f64; 3];
    let mut delta = [0.0f64; 3];
    for j in 0..3 {
        let to = if step[j] > 0 { p[j] as f64 + 1.0 - start[j] } else { start[j] - p[j] as f64 };
        let d = dir[j].abs().max(f64::EPSILON);
        lengths[j] = to / d;
        delta[j] = 1.0 / d;
    }
    let nearest = |l: &[f64; 3]| if l[0] < l[1] && l[0] < l[2] { 0 } else if l[1] < l[2] { 1 } else { 2 };
    let mut ax = nearest(&lengths);
    if visit(p, 0.0, lengths[ax], max_length) {
        return;
    }
    while lengths[ax] < max_length {
        p[ax] += step[ax];
        let in_length = lengths[ax];
        lengths[ax] += delta[ax];
        ax = nearest(&lengths);
        if visit(p, in_length, lengths[ax], max_length) {
            break;
        }
    }
}

#[inline]
fn sign(x: f64) -> i64 {
    (x > 0.0) as i64 - (x < 0.0) as i64
}

/// Effective free path `−ln(1 − λ₁z) / λ₁` (`z` when `λ₁ = 0`).
#[inline]
fn eff_free_path(z: f64, lambda1: f64) -> f64 {
    if lambda1 <= 0.0 {
        return z;
    }
    let x = (lambda1 * z).min(1.0 - 1e-9);
    -(-x).ln_1p() / lambda1
}

/// Sums are kept in `f64` while pulses are added in whatever order the threads
/// get to them, so that the `f32` results do not depend on that order.
#[inline]
fn fadd(a: &AtomicU64, v: f64) {
    let mut cur = a.load(Relaxed);
    loop {
        let new = (f64::from_bits(cur) + v).to_bits();
        match a.compare_exchange_weak(cur, new, Relaxed, Relaxed) {
            Ok(_) => return,
            Err(c) => cur = c,
        }
    }
}

/// Everything a pulse adds to one voxel, side by side: a traversal touches
/// most fields of each voxel it visits, so a voxel should be one stretch of
/// memory rather than an entry in twenty arrays.
pub(crate) struct Cell {
    f: [AtomicU64; F::COUNT],
    i: [AtomicI32; I::COUNT],
}

/// Shared accumulators, written concurrently.
struct Accum {
    cells: Vec<Cell>,
    sub: Vec<AtomicU8>,
}

impl Accum {
    #[inline]
    fn addf(&self, field: F, idx: usize, v: f64) {
        fadd(&self.cells[idx].f[field as usize], v);
    }

    #[inline]
    fn addi(&self, field: I, idx: usize) {
        self.cells[idx].i[field as usize].fetch_add(1, Relaxed);
    }
}

#[derive(Clone, Copy)]
struct Ret {
    pos: Point,
    range: f64,
    bound: bool,
    intensity: f64,
    foliage: u8,
}

/// One intercepted beam for the exact PPL solve.
struct PplHit {
    voxel: usize,
    chord: f32,
    section: f32,
}

struct Tracer<'a> {
    geom: &'a Geom,
    acc: &'a Accum,
    params: &'a VoxelParams,
    peaks: Option<&'a [f64]>,
    ground: Option<&'a [f64]>,
    tan_half_div: f64,
    beam_diameter: f64,
    lambda1: f64,
    ppl: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// Whole pulse, unit weight: counts and potential path length.
    Full,
    /// One echo segment carrying `weight` of the pulse.
    Segment,
    /// `Full` and `Segment` in one walk, for a pulse with a single segment.
    Both,
    Occluded,
}

struct Ray {
    vox_start: Point,
    vox_dir: Point,
    beam_origin: Point,
    unbound: bool,
    foliage: u8,
    weight: f64,
}

impl Tracer<'_> {
    fn section(&self, origin: &Point, cell: [i64; 3]) -> f64 {
        let r = self.tan_half_div * norm(&sub(&self.geom.center(cell), origin)) + 0.5 * self.beam_diameter;
        PI * r * r
    }

    fn walk(&self, start: &Point, end: &Point, beam_origin: &Point, pass: Pass, weight: f64, unbound: bool, fol: u8) {
        let (mut cs, mut ce) = (*start, *end);
        if !self.geom.clip(&mut cs, &mut ce, 1e-10) {
            return;
        }
        let (vs, ve) = (self.geom.to_vox(&cs), self.geom.to_vox(&ce));
        let d = sub(&ve, &vs);
        let len = norm(&d);
        if !(len > 0.0) {
            return;
        }
        let ray = Ray { vox_start: vs, vox_dir: scale(&d, 1.0 / len), beam_origin: *beam_origin, unbound, foliage: fol, weight };
        let zenith = ray.vox_dir[2].clamp(-1.0, 1.0).acos();
        let az = ray.vox_dir[0].atan2(ray.vox_dir[1]);
        let (sin_az, cos_az) = az.sin_cos();
        walk_grid(&vs, &ve, |cell, in_len, out_len, max_len| {
            self.visit(&ray, pass, zenith, sin_az, cos_az, cell, in_len, out_len, max_len);
            false
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn visit(&self, ray: &Ray, pass: Pass, zenith: f64, sin_az: f64, cos_az: f64, cell: [i64; 3], mut in_len: f64, out_len: f64, max_len: f64) {
        let Some(idx) = self.geom.flat(cell) else { return };
        let end_len = out_len.min(max_len);
        let column = cell[0] as usize + self.geom.shape[0] * cell[1] as usize;

        if pass != Pass::Occluded {
            if let Some(peaks) = self.peaks {
                let peak = peaks[column];
                let in_h = ray.vox_start[2] + ray.vox_dir[2] * in_len;
                let end_h = ray.vox_start[2] + ray.vox_dir[2] * end_len;
                if ray.vox_dir[2] < 0.0 && in_h > peak && end_h <= peak {
                    let t = (in_h - peak) / (in_h - end_h);
                    in_len += (end_len - in_len) * t.clamp(0.0, 0.99);
                }
            }
        }
        let size = self.geom.size;
        let length = (end_len - in_len) * size;
        let chord = (out_len - in_len) * size;
        let exits = end_len >= out_len - 1e-6;
        let a = self.acc;

        if pass == Pass::Occluded {
            if let Some(g) = self.ground {
                if self.geom.center(cell)[2] < g[column] {
                    return;
                }
            }
            a.addi(I::NumRaysOccluded, idx);
            a.addf(F::PathLengthOccluded, idx, length * ray.weight);
            return;
        }
        if pass != Pass::Segment {
            a.addi(I::NumBeams, idx);
            a.addf(F::PathLength, idx, chord);
            a.addf(F::PathLengthSq, idx, chord * chord);
            let split = self.params.subvoxel_split;
            if split > 0 {
                let local = |l: f64| {
                    let mut p = [0.0; 3];
                    for k in 0..3 {
                        p[k] = (ray.vox_start[k] + ray.vox_dir[k] * l - cell[k] as f64) * split as f64;
                    }
                    p
                };
                let mut bits = 0u64;
                let s = split as i64;
                walk_grid(&local(in_len), &local(end_len), |c, _, _, _| {
                    if (0..3).all(|k| c[k] >= 0 && c[k] < s) {
                        bits |= 1 << (c[0] + s * (c[1] + s * c[2]));
                    }
                    false
                });
                let n_sub = split * split * split;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let _ = a.sub[idx * n_sub + b].fetch_update(Relaxed, Relaxed, |v| v.checked_add(1));
                }
            }
            if exits {
                if !ray.unbound {
                    a.addi(I::NumMissRays, idx);
                }
                a.addf(F::SumMissDelta, idx, chord);
            }
        }
        if pass != Pass::Full {
            let w = ray.weight;
            a.addf(F::NumBeamsWeighted, idx, w);
            a.addf(F::FreePathLength, idx, length * w);
            match ray.foliage {
                foliage::PLANT => a.addf(F::FreePathLengthPlant, idx, length * w),
                foliage::LEAF => a.addf(F::FreePathLengthLeaf, idx, length * w),
                foliage::WOOD => a.addf(F::FreePathLengthWood, idx, length * w),
                _ => {}
            }
            let eff = eff_free_path(length, self.lambda1);
            a.addf(F::EffectiveFreePathLength, idx, eff * w);
            a.addf(F::SumOfAngles, idx, zenith * w);
            a.addf(F::SumSinAzimuth, idx, sin_az * w);
            a.addf(F::SumCosAzimuth, idx, cos_az * w);
            // Distance from the scanner, not from the previous echo (as AMAPVox).
            a.addf(F::SumOfLaserDistances, idx, norm(&sub(&self.geom.center(cell), &ray.beam_origin)) * w);
            if self.params.beam.is_some() {
                let bs = self.section(&ray.beam_origin, cell) * w;
                a.addf(F::BsEntering, idx, bs);
                a.addf(F::BsFreePath, idx, bs * length);
                a.addf(F::BsEffectiveFreePath, idx, bs * eff);
                if exits {
                    a.addf(F::BsPotential, idx, bs);
                }
            }
            if ray.unbound {
                a.addi(I::NumUnboundRays, idx);
                a.addf(F::PathLengthUnbound, idx, length * w);
            }
        }
    }

    fn echo_weights(&self, rets: &[Ret], w: &mut Vec<f32>) {
        let n = rets.len();
        w.clear();
        w.resize(n, 0.0);
        let equal = |w: &mut Vec<f32>| w.iter_mut().for_each(|v| *v = 1.0 / n as f32);
        match self.params.weighting {
            WeightMethod::Equal => equal(w),
            WeightMethod::Full => w[n - 1] = 1.0,
            // A pulse whose first return is a miss or the ground still samples
            // every voxel it crosses; only its hit is not counted.
            WeightMethod::First => w[0] = 1.0,
            WeightMethod::Relative => {
                let sum: f64 = rets.iter().map(|r| r.intensity).sum();
                if sum > 0.0 {
                    for (v, r) in w.iter_mut().zip(rets) {
                        *v = (r.intensity / sum) as f32;
                    }
                } else {
                    equal(w);
                }
            }
            WeightMethod::Strongest => {
                let mut best = 0;
                for k in 1..n {
                    if rets[k].intensity > rets[best].intensity {
                        best = k;
                    }
                }
                w[best] = 1.0;
            }
        }
    }

    /// All echoes of one pulse, nearest first.
    fn process(&self, origin: &Point, rets: &[Ret], echo_w: &mut Vec<f32>, seg_w: &mut Vec<f32>, ppl_hits: &mut Vec<PplHit>) {
        let n = rets.len();
        if n == 0 {
            return;
        }
        self.echo_weights(rets, echo_w);
        // Fraction of the pulse still travelling along each segment.
        seg_w.clear();
        seg_w.resize(n, 0.0);
        let mut run = 0.0f32;
        for k in (0..n).rev() {
            run += echo_w[k];
            seg_w[k] = run;
        }
        let last_seg = (0..n).rev().find(|&k| seg_w[k] > 0.0);
        let far = &rets[n - 1];
        let ray_vec = sub(&far.pos, origin);
        let ray_len = norm(&ray_vec);

        if let Some(last) = last_seg {
            if n == 1 && ray_len >= 1e-6 {
                // The whole pulse is its only segment: one walk does both passes.
                self.walk(origin, &far.pos, origin, Pass::Both, seg_w[0] as f64, !far.bound, far.foliage);
            } else if ray_len >= 1e-6 {
                self.walk(origin, &rets[last].pos, origin, Pass::Full, 1.0, !rets[last].bound, 0);
            }
            let mut seg_start = *origin;
            for (k, r) in rets.iter().enumerate().take(if n == 1 { 0 } else { last + 1 }) {
                if norm(&sub(&r.pos, &seg_start)) >= 1e-6 {
                    self.walk(&seg_start, &r.pos, origin, Pass::Segment, seg_w[k] as f64, !r.bound, r.foliage);
                }
                seg_start = r.pos;
            }
        }

        let dir = (ray_len > 1e-12).then(|| scale(&ray_vec, 1.0 / ray_len));
        let a = self.acc;
        for (k, r) in rets.iter().enumerate() {
            if !r.bound || seg_w[k] <= 0.0 {
                continue;
            }
            let cell = self.geom.cell_of(&r.pos);
            let Some(idx) = self.geom.flat(cell) else { continue };
            // Full chord of the hit voxel and the free path from its entry to the echo.
            let (mut chord, mut free) = (0.0, 0.0);
            if let Some(d) = dir {
                let (mut t_in, mut t_out) = (0.0f64, 1e30f64);
                for ax in 0..3 {
                    if d[ax].abs() > 1e-15 {
                        let lo = self.geom.origin[ax] + cell[ax] as f64 * self.geom.size;
                        let t1 = (lo - origin[ax]) / d[ax];
                        let t2 = (lo + self.geom.size - origin[ax]) / d[ax];
                        t_in = t_in.max(t1.min(t2));
                        t_out = t_out.min(t1.max(t2));
                    }
                }
                chord = (t_out - t_in).max(0.0);
                free = (r.range - t_in).min(chord).max(0.0);
            }
            a.addi(I::NumHits, idx);
            a.addf(F::HitsWeighted, idx, echo_w[k] as f64);
            a.addf(F::SumHitDelta, idx, chord);
            match r.foliage {
                foliage::LEAF => a.addi(I::NumHitLeaf, idx),
                foliage::WOOD => a.addi(I::NumHitWood, idx),
                _ => {}
            }
            if r.foliage != foliage::EXCLUDED {
                a.addi(I::NumHitPlant, idx);
            }
            if self.params.beam.is_some() {
                // This echo's own share of the pulse, not the cumulative segment weight.
                let bs = self.section(origin, cell) * echo_w[k] as f64;
                a.addf(F::BsIntercepted, idx, bs);
                a.addf(F::BsEffFreePathHits, idx, bs * eff_free_path(free, self.lambda1));
            }
        }

        if self.params.occlusion && far.bound && ray_len >= 1e-6 {
            let mut diag = 0.0;
            for k in 0..3 {
                diag += (self.geom.shape[k] as f64 * self.geom.size).powi(2);
            }
            let end = add(&far.pos, &scale(&ray_vec, 2.0 * diag.sqrt() / ray_len));
            self.walk(&far.pos, &end, origin, Pass::Occluded, 1.0, false, 0);
        }

        if self.ppl {
            self.accumulate_ppl(origin, rets, echo_w, ppl_hits);
        }
    }

    /// Exact PPL: one record per intercepted echo (potential path = full
    /// chord), and the leaving fraction × section × chord per voxel.
    fn accumulate_ppl(&self, origin: &Point, rets: &[Ret], echo_w: &[f32], out: &mut Vec<PplHit>) {
        let (mut cs, mut ce) = (*origin, rets[rets.len() - 1].pos);
        if !self.geom.clip(&mut cs, &mut ce, 1e-10) {
            return;
        }
        let mut f_in = 1.0f64;
        walk_grid(&self.geom.to_vox(&cs), &self.geom.to_vox(&ce), |cell, in_len, out_len, _| {
            let Some(idx) = self.geom.flat(cell) else { return false };
            let chord = (out_len - in_len) * self.geom.size;
            if chord <= 0.0 {
                return false;
            }
            let section = if self.params.beam.is_some() { self.section(origin, cell) } else { 1.0 };
            let mut intercepted = 0.0;
            for (r, &w) in rets.iter().zip(echo_w) {
                if r.bound && self.geom.cell_of(&r.pos) == cell {
                    intercepted += w as f64;
                    out.push(PplHit { voxel: idx, chord: chord as f32, section: (w as f64 * section) as f32 });
                }
            }
            let leaving = f_in - intercepted;
            if leaving > 0.0 {
                self.acc.addf(F::PplMissWl, idx, leaving * section * chord);
            }
            f_in -= intercepted;
            f_in < 1e-6
        });
    }
}

/// Per voxel, solve `Σ_hits bs·L / (e^{kL} − 1) = ppl_miss_wl` for `k`
/// (AMAPVox's exact PPL maximum-likelihood estimate, capped at 20 m⁻¹).
fn solve_ppl(hits: Vec<PplHit>, miss: &[f32], n: usize) -> Vec<f32> {
    const K_MAX: f64 = 20.0;
    const EPS: f64 = 1e-12;
    let mut by_voxel: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
    for h in hits {
        by_voxel.entry(h.voxel).or_default().push((h.chord as f64, h.section as f64));
    }
    let solved: Vec<(usize, f32)> = by_voxel
        .par_iter()
        .map(|(&idx, recs)| {
            let target = miss[idx] as f64;
            let f = |k: f64| recs.iter().map(|&(l, bs)| { let e = (k * l).exp() - 1.0; if e > EPS { bs * l / e } else { 0.0 } }).sum::<f64>();
            // No beam left the voxel, or the root lies beyond the cap.
            if target <= EPS || f(K_MAX) >= target {
                return (idx, K_MAX as f32);
            }
            let (mut lo, mut hi) = (EPS, K_MAX);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if f(mid) > target { lo = mid } else { hi = mid }
            }
            (idx, (0.5 * (lo + hi)) as f32)
        })
        .collect();
    let mut out = vec![-1.0f32; n];
    for (idx, k) in solved {
        out[idx] = k;
    }
    out
}

/// Accumulators for one grid, filled by any number of [`Engine::add`] calls.
pub(crate) struct Engine {
    pub geom: Geom,
    pub params: VoxelParams,
    acc: Accum,
    hits: Vec<PplHit>,
    ppl: bool,
}

impl Engine {
    pub fn new(params: &VoxelParams, geom: Geom) -> Self {
        let n = geom.shape[0] * geom.shape[1] * geom.shape[2];
        let ppl = params.attenuation.contains(&Attenuation::Ppl);
        let n_sub = params.subvoxel_split.pow(3);
        let acc = Accum {
            cells: std::iter::repeat_with(|| Cell { f: std::array::from_fn(|_| AtomicU64::new(0)), i: std::array::from_fn(|_| AtomicI32::new(0)) }).take(n).collect(),
            sub: std::iter::repeat_with(|| AtomicU8::new(0)).take(n * n_sub).collect(),
        };
        Engine { geom, params: params.clone(), acc, hits: Vec::new(), ppl }
    }

    pub fn add(&mut self, inputs: &VoxelInputs, peaks: Option<&[f64]>, ground: Option<&[f64]>) {
        let params = &self.params;
        let geom = &self.geom;
        let beam = params.beam.unwrap_or(super::BeamSpec { diameter: 0.0, divergence: 0.0 });
        let tracer = Tracer {
            geom,
            acc: &self.acc,
            params,
            peaks,
            ground,
            tan_half_div: (0.5 * beam.divergence).tan(),
            beam_diameter: beam.diameter,
            lambda1: if params.average_leaf_area > 0.0 { 0.25 * params.average_leaf_area / params.voxel_size.powi(3) } else { 0.0 },
            ppl: self.ppl,
        };
        let shots = inputs.shots;
        struct Local {
            rets: Vec<Ret>,
            echo_w: Vec<f32>,
            seg_w: Vec<f32>,
            hits: Vec<PplHit>,
            done: u64,
        }
        let task = crate::progress::start("tracing pulses", shots.n_shots() as u64);
        let mut hits: Vec<PplHit> = (0..shots.n_shots())
            .into_par_iter()
            .with_min_len(256)
            .fold(
                || Local { rets: Vec::new(), echo_w: Vec::new(), seg_w: Vec::new(), hits: Vec::new(), done: 0 },
                |mut l, s| {
                    let (o, d) = (shots.origin[s], shots.direction[s]);
                    let first = shots.echo_start[s];
                    l.rets.clear();
                    for e in first..first + shots.echo_count[s] as usize {
                        let range = shots.echo_range[e];
                        let is_ground = inputs.ground.is_some_and(|g| g[e]);
                        l.rets.push(Ret {
                            pos: add(&o, &scale(&d, range)),
                            range,
                            bound: !is_ground,
                            intensity: inputs.intensity.map_or(0.0, |v| v[e]),
                            foliage: if is_ground { foliage::EXCLUDED } else { inputs.foliage.map_or(foliage::PLANT, |f| f[e]) },
                        });
                    }
                    if l.rets.is_empty() {
                        let end = unbounded_end(&o, &d, geom, params.unbounded_range);
                        l.rets.push(Ret { pos: end, range: norm(&sub(&end, &o)), bound: false, intensity: 0.0, foliage: foliage::EXCLUDED });
                    }
                    tracer.process(&o, &l.rets, &mut l.echo_w, &mut l.seg_w, &mut l.hits);
                    l.done += 1;
                    if l.done % 4096 == 0 {
                        task.inc(4096);
                    }
                    l
                },
            )
            .map(|l| l.hits)
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        self.hits.append(&mut hits);
    }

    pub fn finish(self) -> RayVoxels {
        let n = self.geom.shape[0] * self.geom.shape[1] * self.geom.shape[2];
        let n_sub = self.params.subvoxel_split.pow(3);
        // Unused groups stay empty (they read as 0).
        let cells = &self.acc.cells;
        let f: Vec<Vec<f32>> = F::ALL
            .iter()
            .map(|fld| {
                let on = if fld.is_beam() { self.params.beam.is_some() } else if *fld == F::PplMissWl { self.ppl } else { true };
                if on { cells.par_iter().map(|c| f64::from_bits(c.f[*fld as usize].load(Relaxed)) as f32).collect() } else { Vec::new() }
            })
            .collect();
        let i: Vec<Vec<i32>> = I::ALL.iter().map(|fld| cells.par_iter().map(|c| c.i[*fld as usize].load(Relaxed)).collect()).collect();
        let ppl_lambda = self.ppl.then(|| solve_ppl(self.hits, &f[F::PplMissWl as usize], n));
        RayVoxels {
            origin: self.geom.origin,
            voxel_size: self.geom.size,
            shape: self.geom.shape,
            params: self.params,
            has_leaf: false,
            has_wood: false,
            f,
            i,
            ppl_lambda,
            subvoxel_counts: (n_sub > 0).then(|| self.acc.sub.into_iter().map(AtomicU8::into_inner).collect()),
            ground_height: None,
            wood_volume: None,
            predominant_tree: None,
            tree_iad: BTreeMap::new(),
        }
    }
}
