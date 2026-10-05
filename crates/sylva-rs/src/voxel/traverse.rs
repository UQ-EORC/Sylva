// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
// see THIRD_PARTY_NOTICES.md.
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

/// A box of voxels, `lo` to `lo + shape` (exclusive), that one engine
/// accumulates: a block of a grid traced block by block. Pulses are still
/// walked through the whole grid, so every voxel of the box receives exactly
/// the contributions (and in the same order) that a whole-grid trace gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Window {
    pub lo: [usize; 3],
    pub shape: [usize; 3],
}

impl Window {
    #[inline]
    pub fn local(&self, p: [i64; 3]) -> Option<usize> {
        let mut q = [0usize; 3];
        for k in 0..3 {
            let d = p[k] - self.lo[k] as i64;
            if d < 0 || d as usize >= self.shape[k] {
                return None;
            }
            q[k] = d as usize;
        }
        Some(q[0] + self.shape[0] * (q[1] + self.shape[1] * q[2]))
    }

    pub fn n_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Whether the segment `a`-`b` (voxel units) passes within `margin` of the box.
    pub fn touches(&self, a: &Point, b: &Point, margin: f64) -> bool {
        let (mut s0, mut s1) = (0.0f64, 1.0f64);
        for k in 0..3 {
            let (lo, hi) = (self.lo[k] as f64 - margin, (self.lo[k] + self.shape[k]) as f64 + margin);
            let d = b[k] - a[k];
            if d == 0.0 {
                if a[k] < lo || a[k] > hi {
                    return false;
                }
                continue;
            }
            let (t0, t1) = ((lo - a[k]) / d, (hi - a[k]) / d);
            s0 = s0.max(t0.min(t1));
            s1 = s1.min(t0.max(t1));
            if s0 > s1 {
                return false;
            }
        }
        true
    }
}

/// How far short of `end` a segment stops, so that one ending exactly on a
/// voxel face does not reach into the voxel beyond it.
const FACE_TOLERANCE: f64 = 1e-10;

/// The voxels a segment crosses, in order: the traversal of Amanatides & Woo
/// (1987). Coordinates are in voxel units, voxel `i` spanning `[i, i + 1)` on
/// each axis. `visit(cell, t_enter, t_exit, length)` receives the distances
/// along the segment at which it enters and leaves `cell` (`t_exit` can pass
/// `length` in the last voxel) and returns `true` to stop.
///
/// `visit: impl FnMut(...) -> bool` is a callback: the caller passes a closure
/// (a lambda), which may hold and change state of its own between calls —
/// several below accumulate into a counter or a bit mask. Writing it this way
/// rather than returning a list of voxels means nothing is allocated per
/// pulse, which matters when there are hundreds of millions of them.
pub(crate) fn walk_grid(start: &Point, end: &Point, mut visit: impl FnMut([i64; 3], f64, f64, f64) -> bool) {
    let span = sub(end, start);
    let full = norm(&span);
    if !(full > 0.0) {
        return;
    }
    let length = full - FACE_TOLERANCE;
    // Per axis (the paper's X, Y, Z): the voxel, the step direction, the
    // distance to the first boundary (tMax) and between boundaries (tDelta).
    let mut cell = [0i64; 3];
    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for axis in 0..3 {
        let x = start[axis];
        cell[axis] = x.floor() as i64;
        let cosine = span[axis] / full;
        if cosine > 0.0 {
            step[axis] = 1;
            t_delta[axis] = 1.0 / cosine;
            t_max[axis] = ((cell[axis] + 1) as f64 - x) / cosine;
        } else if cosine < 0.0 {
            step[axis] = -1;
            t_delta[axis] = -1.0 / cosine;
            t_max[axis] = (x - cell[axis] as f64) / -cosine;
        }
    }
    let mut t_enter = 0.0;
    loop {
        // The boundary met first; ties go to the later axis, as in the paper.
        let axis = if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] { 0 } else { 2 }
        } else if t_max[1] < t_max[2] {
            1
        } else {
            2
        };
        if visit(cell, t_enter, t_max[axis], length) || t_max[axis] >= length {
            return;
        }
        t_enter = t_max[axis];
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
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
///
/// Reading this without Rust: there is no atomic `f64`, so the running sum is
/// kept as the `f64`'s own bit pattern inside an atomic 64-bit integer
/// (`to_bits` / `from_bits` reinterpret, they do not convert). The loop is the
/// usual lock-free update: read the current value, work out what it should
/// become, and swap it in *only if* no other thread has changed it meanwhile.
/// `compare_exchange_weak` returns the value it actually found when that
/// fails, so the loop simply tries again from there. `Relaxed` says we need
/// nothing of the ordering between threads beyond the single value being
/// updated atomically, which is true here: every sum stands alone.
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
///
/// The fields are fixed-size arrays indexed by the field enums [`F`] (sums,
/// held as bits — see [`fadd`]) and [`I`] (counts), so `F::PathLength as usize`
/// is simply a slot number.
pub(crate) struct Cell {
    f: [AtomicU64; F::COUNT],
    i: [AtomicI32; I::COUNT],
}

/// Shared accumulators, written concurrently.
///
/// Threads trace different pulses into the same grid at the same time. Rather
/// than lock a voxel, every slot is an atomic the hardware updates on its own,
/// which is why `addf` and `addi` below take `&self` (a shared reference) and
/// not `&mut self`: nothing here is exclusively borrowed, so any number of
/// threads may hold it at once.
struct Accum {
    cells: Vec<Cell>,
    /// Sub-voxel occupancy, `n_sub` slots per voxel laid end to end, so voxel
    /// `idx`'s slots start at `idx * n_sub`.
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
pub(crate) struct Ret {
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
    /// Only this box is accumulated (indices are then local to it).
    window: Option<&'a Window>,
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

/// Margin (voxels) within which a segment counts as reaching a window.
const WINDOW_MARGIN: f64 = 1e-6;

impl Tracer<'_> {
    /// Index of a voxel in the accumulators: flat in the grid, or in the window.
    #[inline]
    fn index(&self, cell: [i64; 3]) -> Option<usize> {
        match self.window {
            None => self.geom.flat(cell),
            Some(w) => w.local(cell),
        }
    }

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
        if self.window.is_some_and(|w| !w.touches(&vs, &ve, WINDOW_MARGIN)) {
            return;
        }
        let ray = Ray { vox_start: vs, vox_dir: scale(&d, 1.0 / len), beam_origin: *beam_origin, unbound, foliage: fol, weight };
        let zenith = ray.vox_dir[2].clamp(-1.0, 1.0).acos();
        let az = ray.vox_dir[0].atan2(ray.vox_dir[1]);
        let (sin_az, cos_az) = az.sin_cos();
        // The walk is the whole-grid one; with a window it stops once it has
        // left the box (every coordinate is monotone, so it cannot come back).
        let mut inside = false;
        walk_grid(&vs, &ve, |cell, in_len, out_len, max_len| {
            // `index` gives `None` for a voxel outside the grid (or outside the
            // window being accumulated); `let ... else` takes the value when
            // there is one and otherwise runs the block, which here ends the
            // walk once the ray has crossed the window and left it again.
            let Some(idx) = self.index(cell) else { return inside && self.window.is_some() };
            inside = true;
            self.visit(&ray, pass, zenith, sin_az, cos_az, cell, idx, in_len, out_len, max_len);
            false
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn visit(&self, ray: &Ray, pass: Pass, zenith: f64, sin_az: f64, cos_az: f64, cell: [i64; 3], idx: usize, mut in_len: f64, out_len: f64, max_len: f64) {
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
                // Which of the voxel's own sub-cells this pulse crossed. A
                // sub-cell is counted once however long the pulse spent in it,
                // so the crossing is recorded as one bit per sub-cell in a
                // 64-bit mask (`split` is at most 4, so at most 64 of them).
                let mut bits = 0u64;
                let s = split as i64;
                walk_grid(&local(in_len), &local(end_len), |c, _, _, _| {
                    if (0..3).all(|k| c[k] >= 0 && c[k] < s) {
                        bits |= 1 << (c[0] + s * (c[1] + s * c[2]));
                    }
                    false
                });
                let n_sub = split * split * split;
                // Walk the bits that are set: `trailing_zeros` is the lowest
                // one, and `bits &= bits - 1` clears it, so the loop runs once
                // per crossed sub-cell rather than once per sub-cell.
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    // The counters are single bytes, so a sub-cell crossed more
                    // than 255 times stops counting rather than wrapping to 0:
                    // `checked_add` gives `None` at the ceiling and
                    // `fetch_update` then leaves the value alone.
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
            let Some(idx) = self.index(cell) else { continue };
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
        let (vs, ve) = (self.geom.to_vox(&cs), self.geom.to_vox(&ce));
        if self.window.is_some_and(|w| !w.touches(&vs, &ve, WINDOW_MARGIN)) {
            return;
        }
        // The share of the pulse still travelling depends on every voxel
        // before this one, so a window walks (without recording) from the start.
        let mut f_in = 1.0f64;
        let mut inside = false;
        walk_grid(&vs, &ve, |cell, in_len, out_len, _| {
            if self.geom.flat(cell).is_none() {
                return false;
            }
            let idx = self.index(cell);
            if idx.is_none() && inside {
                return true;
            }
            inside |= idx.is_some();
            let chord = (out_len - in_len) * self.geom.size;
            if chord <= 0.0 {
                return false;
            }
            let section = if self.params.beam.is_some() { self.section(origin, cell) } else { 1.0 };
            let mut intercepted = 0.0;
            for (r, &w) in rets.iter().zip(echo_w) {
                if r.bound && self.geom.cell_of(&r.pos) == cell {
                    intercepted += w as f64;
                    if let Some(idx) = idx {
                        out.push(PplHit { voxel: idx, chord: chord as f32, section: (w as f64 * section) as f32 });
                    }
                }
            }
            if let Some(idx) = idx {
                let leaving = f_in - intercepted;
                if leaving > 0.0 {
                    self.acc.addf(F::PplMissWl, idx, leaving * section * chord);
                }
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

/// The echoes of pulse `s` as traced (a pulse without one gets a far end).
fn pulse_rets(inputs: &VoxelInputs, s: usize, geom: &Geom, params: &VoxelParams, rets: &mut Vec<Ret>) {
    let shots = inputs.shots;
    let (o, d) = (shots.origin[s], shots.direction[s]);
    let first = shots.echo_start[s];
    rets.clear();
    for e in first..first + shots.echo_count[s] as usize {
        let range = shots.echo_range[e];
        let is_ground = inputs.ground.is_some_and(|g| g[e]);
        rets.push(Ret {
            pos: add(&o, &scale(&d, range)),
            range,
            bound: !is_ground,
            intensity: inputs.intensity.map_or(0.0, |v| v[e]),
            foliage: if is_ground { foliage::EXCLUDED } else { inputs.foliage.map_or(foliage::PLANT, |f| f[e]) },
        });
    }
    if rets.is_empty() {
        let end = unbounded_end(&o, &d, geom, params.unbounded_range);
        rets.push(Ret { pos: end, range: norm(&sub(&end, &o)), bound: false, intensity: 0.0, foliage: foliage::EXCLUDED });
    }
}

/// The segment of pulse `s` that any of its walks can reach, in voxel
/// units: from its origin to its farthest echo, or to the end of the
/// occlusion ray beyond it. `rets` is scratch space.
pub(crate) fn pulse_extent(inputs: &VoxelInputs, s: usize, geom: &Geom, params: &VoxelParams, rets: &mut Vec<Ret>) -> (Point, Point) {
    pulse_rets(inputs, s, geom, params, rets);
    let o = inputs.shots.origin[s];
    let far = &rets[rets.len() - 1];
    let ray_vec = sub(&far.pos, &o);
    let ray_len = norm(&ray_vec);
    let mut ends: Vec<Point> = rets.iter().map(|r| r.pos).collect();
    if params.occlusion && far.bound && ray_len >= 1e-6 {
        let mut diag = 0.0;
        for k in 0..3 {
            diag += (geom.shape[k] as f64 * geom.size).powi(2);
        }
        ends.push(add(&far.pos, &scale(&ray_vec, 2.0 * diag.sqrt() / ray_len)));
    }
    // Every walk lies on the line through the origin; take its span.
    let dir = if ray_len > 0.0 { scale(&ray_vec, 1.0 / ray_len) } else { inputs.shots.direction[s] };
    let (mut t0, mut t1) = (0.0f64, 0.0f64);
    for p in &ends {
        let v = sub(p, &o);
        let t = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
        t0 = t0.min(t);
        t1 = t1.max(t);
    }
    (geom.to_vox(&add(&o, &scale(&dir, t0))), geom.to_vox(&add(&o, &scale(&dir, t1))))
}

/// Reusable per-pulse buffers.
#[derive(Default)]
pub(crate) struct Scratch {
    pub rets: Vec<Ret>,
    echo_w: Vec<f32>,
    seg_w: Vec<f32>,
}

/// Accumulators for one grid, filled by any number of [`Engine::add`] calls.
pub(crate) struct Engine {
    pub geom: Geom,
    pub params: VoxelParams,
    /// Only this box of the grid is accumulated.
    window: Option<Window>,
    acc: Accum,
    hits: Vec<PplHit>,
    ppl: bool,
}

impl Engine {
    pub fn new(params: &VoxelParams, geom: Geom) -> Self {
        Self::build(params, geom, None)
    }

    /// Accumulators for the voxels of `window` only; pulses are traced
    /// through the whole of `geom` so that the box gets exactly what a
    /// whole-grid trace would give it.
    pub fn new_window(params: &VoxelParams, geom: Geom, window: Window) -> Self {
        Self::build(params, geom, Some(window))
    }

    fn build(params: &VoxelParams, geom: Geom, window: Option<Window>) -> Self {
        let n = window.map_or(geom.shape[0] * geom.shape[1] * geom.shape[2], |w| w.n_voxels());
        let ppl = params.attenuation.contains(&Attenuation::Ppl);
        let n_sub = params.subvoxel_split.pow(3);
        let acc = Accum {
            cells: std::iter::repeat_with(|| Cell { f: std::array::from_fn(|_| AtomicU64::new(0)), i: std::array::from_fn(|_| AtomicI32::new(0)) }).take(n).collect(),
            sub: std::iter::repeat_with(|| AtomicU8::new(0)).take(n * n_sub).collect(),
        };
        Engine { geom, params: params.clone(), window, acc, hits: Vec::new(), ppl }
    }

    fn tracer<'a>(&'a self, peaks: Option<&'a [f64]>, ground: Option<&'a [f64]>) -> Tracer<'a> {
        let params = &self.params;
        let beam = params.beam.unwrap_or(super::BeamSpec { diameter: 0.0, divergence: 0.0 });
        Tracer {
            geom: &self.geom,
            window: self.window.as_ref(),
            acc: &self.acc,
            params,
            peaks,
            ground,
            tan_half_div: (0.5 * beam.divergence).tan(),
            beam_diameter: beam.diameter,
            lambda1: if params.average_leaf_area > 0.0 { 0.25 * params.average_leaf_area / params.voxel_size.powi(3) } else { 0.0 },
            ppl: self.ppl,
        }
    }

    pub fn add(&mut self, inputs: &VoxelInputs, peaks: Option<&[f64]>, ground: Option<&[f64]>) {
        let params = &self.params;
        let geom = &self.geom;
        let tracer = self.tracer(peaks, ground);
        let shots = inputs.shots;
        struct Local {
            scratch: Scratch,
            hits: Vec<PplHit>,
            done: u64,
        }
        let task = crate::util::progress::start("tracing pulses", shots.n_shots() as u64);
        // Trace every pulse, on as many threads as the machine has. Reading
        // this without Rust: `into_par_iter` turns the range of pulse numbers
        // into a parallel loop (rayon), `with_min_len(256)` hands out work in
        // chunks of at least that many so the bookkeeping does not cost more
        // than the tracing, and `fold` gives each thread its own `Local` —
        // reusable buffers, and the hits it finds — so the threads share
        // nothing but the grid's atomics. `reduce` then joins those per-thread
        // lists into one. The grid itself is written as the tracing goes, by
        // the atomic adds in [`Accum`]; only the PPL hits have to be collected.
        let mut hits: Vec<PplHit> = (0..shots.n_shots())
            .into_par_iter()
            .with_min_len(256)
            .fold(
                || Local { scratch: Scratch::default(), hits: Vec::new(), done: 0 },
                |mut l, s| {
                    let sc = &mut l.scratch;
                    pulse_rets(inputs, s, geom, params, &mut sc.rets);
                    tracer.process(&shots.origin[s], &sc.rets, &mut sc.echo_w, &mut sc.seg_w, &mut l.hits);
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

    /// Trace the pulses `pulses` of `inputs` one after another, in the order given.
    pub fn add_sequential(&mut self, inputs: &VoxelInputs, pulses: &[u32], peaks: Option<&[f64]>, ground: Option<&[f64]>, scratch: &mut Scratch) {
        let tracer = self.tracer(peaks, ground);
        let mut hits = Vec::new();
        for &s in pulses {
            let s = s as usize;
            pulse_rets(inputs, s, &self.geom, &self.params, &mut scratch.rets);
            tracer.process(&inputs.shots.origin[s], &scratch.rets, &mut scratch.echo_w, &mut scratch.seg_w, &mut hits);
        }
        self.hits.append(&mut hits);
    }

    /// Voxels held.
    pub fn n_cells(&self) -> usize {
        self.acc.cells.len()
    }

    pub fn finish(self) -> RayVoxels {
        let (origin, shape) = match self.window {
            None => (self.geom.origin, self.geom.shape),
            Some(w) => (std::array::from_fn(|k| self.geom.origin[k] + w.lo[k] as f64 * self.geom.size), w.shape),
        };
        let n = shape[0] * shape[1] * shape[2];
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
            origin,
            voxel_size: self.geom.size,
            shape,
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
