// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Block tracing: a voxel grid traced a few blocks at a time, so that the
//! accumulators of the whole grid never have to be held at once.
//!
//! The grid is cut into blocks of `block` voxels. The blocks are traced in
//! passes, as many per pass as the memory allowance holds; each pass streams
//! every pulse once, finds the blocks of the pass its line passes through
//! (a slab test, nested axis by axis, grown by 10⁻⁶ voxel), and traces it
//! into those blocks only.
//!
//! A pulse is always walked through the whole grid, from where it enters,
//! with the whole-grid arithmetic: a block only keeps the visits that fall
//! inside it. Every voxel therefore receives exactly the numbers a
//! whole-grid trace adds to it: the chord lengths, the beam sections, the
//! share of the pulse still travelling after the echoes before the block
//! (echo weighting), the potential path length solve's leaving fraction, and
//! the occlusion ray beyond the last echo. Each block is filled by one
//! thread, pulse after pulse in file order, so its double-precision sums are
//! added in the same order whatever the block size or the number of
//! threads: blocked results are identical for any block size and worker
//! count, and identical to a whole-grid trace on one thread. A whole-grid
//! trace on several threads adds in the order the threads reach a voxel,
//! which can move a single-precision result by one unit in the last place.

use std::path::Path;
use std::sync::Mutex;

use rayon::prelude::*;

use super::blocked::{BlockWriter, BlockedGrid};
use super::traverse::{pulse_extent, Engine, Geom, Scratch, Window};
use super::{check_len, column_ground, for_each_file_batch, foliage, grid_shape, iad, pad_bounds, refine, Attenuation, EchoLabels, RayVoxels, VoxelInputs, VoxelParams, WeightMethod};
use crate::error::{Error, Result};
use crate::io::shots::ShotsFile;
use crate::{Raster, Shots};

/// Where the pulses of a blocked trace come from.
pub enum Pulses<'a> {
    /// Pulses in memory with their echo labels.
    Memory(VoxelInputs<'a>),
    /// A shots file, streamed a few row groups at a time on every pass, with
    /// the echo labels taken from its attributes.
    File { file: &'a ShotsFile, labels: &'a EchoLabels },
}

/// How a grid is cut into blocks and how many are traced at once.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockOptions {
    /// Voxels per block along x, y and z (the last block of an axis may be
    /// shorter).
    pub block: [usize; 3],
    /// Bytes the tracing accumulators of one pass may take; `None` is half
    /// of [`crate::util::limits::budget`] (8 GB when the system does not say).
    pub max_memory: Option<u64>,
    /// Threads; 0 uses the global pool.
    pub workers: usize,
}

impl Default for BlockOptions {
    fn default() -> Self {
        BlockOptions { block: [64, 64, 64], max_memory: None, workers: 0 }
    }
}

/// What a blocked trace did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockStats {
    /// Blocks along x, y and z.
    pub n_blocks: [usize; 3],
    /// Passes over the pulses.
    pub n_passes: usize,
    /// Most voxels whose accumulators were held at once.
    pub peak_voxels: usize,
    /// Accumulator bytes per voxel while tracing.
    pub bytes_per_voxel: usize,
    /// Pulses traced into a block, summed over the blocks.
    pub block_pulses: u64,
    /// Blocks written to disk (blocks no pulse reached are not written).
    pub blocks_written: usize,
}

/// Bytes of tracing state per voxel.
pub(crate) fn bytes_per_voxel(params: &VoxelParams) -> usize {
    std::mem::size_of::<super::traverse::Cell>() + params.subvoxel_split.pow(3)
}

/// Pulses in memory are handed out in batches of this many, so that the
/// block lists of one batch stay small.
const MEMORY_BATCH: usize = 1 << 22;

/// Margin (voxels) of the block test: far more than the rounding of any
/// walk, far less than a voxel.
const MARGIN: f64 = 1e-6;

fn validate(params: &VoxelParams, opts: &BlockOptions) -> Result<()> {
    if params.voxel_size.is_nan() || params.voxel_size <= 0.0 {
        return Err(Error::invalid("voxel_size must be positive"));
    }
    if params.subvoxel_split > 4 {
        return Err(Error::invalid("subvoxel_split must be at most 4"));
    }
    if params.attenuation.is_empty() {
        return Err(Error::invalid("at least one attenuation method is needed"));
    }
    if opts.block.contains(&0) {
        return Err(Error::invalid("block sizes must be at least one voxel"));
    }
    Ok(())
}

fn check_inputs(inputs: &VoxelInputs, params: &VoxelParams) -> Result<()> {
    let ne = inputs.shots.n_echoes();
    check_len("ground", inputs.ground, ne)?;
    check_len("foliage", inputs.foliage, ne)?;
    check_len("intensity", inputs.intensity, ne)?;
    check_len("tree_id", inputs.tree_id, ne)?;
    if matches!(params.weighting, WeightMethod::Relative | WeightMethod::Strongest) && inputs.intensity.is_none() {
        return Err(Error::invalid("relative / strongest weighting needs echo intensities"));
    }
    Ok(())
}

/// Hand every pulse of `pulses` to `f` as `(inputs, range)` batches, always
/// in the same order.
fn for_each_batch(pulses: &Pulses, dtm: Option<&Raster>, mut f: impl FnMut(&VoxelInputs, std::ops::Range<usize>) -> Result<()>) -> Result<()> {
    match pulses {
        Pulses::Memory(inputs) => {
            let n = inputs.shots.n_shots();
            for s0 in (0..n).step_by(MEMORY_BATCH) {
                f(inputs, s0..(s0 + MEMORY_BATCH).min(n))?;
            }
            Ok(())
        }
        Pulses::File { file, labels } => for_each_file_batch(file, |shots: Shots| {
            let notes = labels.annotate(&shots, dtm)?;
            let inputs = notes.inputs(&shots, dtm);
            f(&inputs, 0..shots.n_shots())
        }),
    }
}

/// Blocks `[bx, by, bz]` in `blo..bhi` whose box, grown by `m` voxels,
/// the segment `a`-`b` (voxel units) passes through. Axis by axis: the
/// blocks along x the segment spans, then for each the stretch of the
/// segment inside it, the blocks along y that stretch spans, and so on.
fn blocks_on_segment(a: &[f64; 3], b: &[f64; 3], block: [usize; 3], blo: [usize; 3], bhi: [usize; 3], m: f64, out: &mut Vec<[usize; 3]>) {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    // The part of the segment inside the grown box of all the blocks.
    let (mut s0, mut s1) = (0.0f64, 1.0f64);
    for k in 0..3 {
        let (lo, hi) = ((blo[k] * block[k]) as f64 - m, (bhi[k] * block[k]) as f64 + m);
        if !clip_axis(a[k], d[k], lo, hi, &mut s0, &mut s1) {
            return;
        }
    }
    let mut idx = [0usize; 3];
    descend(0, a, &d, s0, s1, block, blo, bhi, m, &mut idx, out);
}

/// Narrow `[s0, s1]` to where `a + s d` lies in `[lo, hi]`; false if empty.
#[inline]
fn clip_axis(a: f64, d: f64, lo: f64, hi: f64, s0: &mut f64, s1: &mut f64) -> bool {
    if d == 0.0 {
        return a >= lo && a <= hi;
    }
    let (t0, t1) = ((lo - a) / d, (hi - a) / d);
    *s0 = s0.max(t0.min(t1));
    *s1 = s1.min(t0.max(t1));
    *s0 <= *s1
}

#[allow(clippy::too_many_arguments)]
fn descend(axis: usize, a: &[f64; 3], d: &[f64; 3], s0: f64, s1: f64, block: [usize; 3], blo: [usize; 3], bhi: [usize; 3], m: f64, idx: &mut [usize; 3], out: &mut Vec<[usize; 3]>) {
    let (x0, x1) = (a[axis] + s0 * d[axis], a[axis] + s1 * d[axis]);
    let (lo, hi) = (x0.min(x1) - m, x0.max(x1) + m);
    let bs = block[axis] as f64;
    let first = ((lo / bs).floor() as i64).max(blo[axis] as i64);
    let last = ((hi / bs).floor() as i64).min(bhi[axis] as i64 - 1);
    for bi in first..=last {
        let (mut t0, mut t1) = (s0, s1);
        if !clip_axis(a[axis], d[axis], bi as f64 * bs - m, (bi + 1) as f64 * bs + m, &mut t0, &mut t1) {
            continue;
        }
        idx[axis] = bi as usize;
        if axis == 2 {
            out.push(*idx);
        } else {
            descend(axis + 1, a, d, t0, t1, block, blo, bhi, m, idx, out);
        }
    }
}

/// One pass: the blocks traced together and where each sits in the pass.
struct Pass {
    windows: Vec<Window>,
    /// Block box of the pass (block indices, `hi` exclusive).
    blo: [usize; 3],
    bhi: [usize; 3],
    /// Position in `windows` of each block of the grid (`u32::MAX` if not in this pass).
    slot: Vec<u32>,
}

/// For each block of the pass, the pulses of `range` that reach it, in order
/// (offsets from `range.start`).
fn bin_pulses(inputs: &VoxelInputs, range: std::ops::Range<usize>, geom: &Geom, params: &VoxelParams, block: [usize; 3], nb: [usize; 3], pass: &Pass) -> Vec<Vec<u32>> {
    const CHUNK: usize = 1 << 14;
    let starts: Vec<usize> = range.clone().step_by(CHUNK).collect();
    let pairs: Vec<Vec<(u32, u32)>> = starts
        .par_iter()
        .map(|&c0| {
            let mut scratch = Scratch::default();
            let mut found = Vec::new();
            let mut out = Vec::new();
            for s in c0..(c0 + CHUNK).min(range.end) {
                let (a, b) = pulse_extent(inputs, s, geom, params, &mut scratch.rets);
                found.clear();
                blocks_on_segment(&a, &b, block, pass.blo, pass.bhi, MARGIN, &mut found);
                for bl in &found {
                    let slot = pass.slot[bl[0] + nb[0] * (bl[1] + nb[1] * bl[2])];
                    if slot != u32::MAX {
                        out.push((slot, (s - range.start) as u32));
                    }
                }
            }
            out
        })
        .collect();
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); pass.windows.len()];
    for chunk in pairs {
        for (slot, s) in chunk {
            lists[slot as usize].push(s);
        }
    }
    lists
}

/// Group the blocks, in order, into passes of at most `budget` voxels.
fn plan_passes(shape: [usize; 3], block: [usize; 3], nb: [usize; 3], budget: usize) -> Result<Vec<Pass>> {
    let mut passes = Vec::new();
    let mut current: Vec<([usize; 3], Window)> = Vec::new();
    let mut held = 0usize;
    let flush = |current: &mut Vec<([usize; 3], Window)>, passes: &mut Vec<Pass>| {
        if current.is_empty() {
            return;
        }
        let mut blo = [usize::MAX; 3];
        let mut bhi = [0usize; 3];
        let mut slot = vec![u32::MAX; nb[0] * nb[1] * nb[2]];
        for (n, (b, _)) in current.iter().enumerate() {
            for k in 0..3 {
                blo[k] = blo[k].min(b[k]);
                bhi[k] = bhi[k].max(b[k] + 1);
            }
            slot[b[0] + nb[0] * (b[1] + nb[1] * b[2])] = n as u32;
        }
        passes.push(Pass { windows: current.drain(..).map(|(_, w)| w).collect(), blo, bhi, slot });
    };
    for bz in 0..nb[2] {
        for by in 0..nb[1] {
            for bx in 0..nb[0] {
                let b = [bx, by, bz];
                let lo: [usize; 3] = std::array::from_fn(|k| b[k] * block[k]);
                let w = Window { lo, shape: std::array::from_fn(|k| block[k].min(shape[k] - lo[k])) };
                let n = w.n_voxels();
                if n > budget {
                    return Err(Error::invalid(format!(
                        "a block of {} x {} x {} voxels needs more than the memory allowed for tracing; use smaller blocks or allow more memory",
                        w.shape[0], w.shape[1], w.shape[2]
                    )));
                }
                if held + n > budget {
                    flush(&mut current, &mut passes);
                    held = 0;
                }
                held += n;
                current.push((b, w));
            }
        }
    }
    flush(&mut current, &mut passes);
    Ok(passes)
}

/// The assembled grid, filled block by block.
struct Assembly {
    vox: RayVoxels,
}

impl Assembly {
    fn put(&mut self, w: &Window, part: &RayVoxels) {
        let [nx, ny, _] = self.vox.shape;
        let [wx, wy, wz] = w.shape;
        let n_sub = self.vox.params.subvoxel_split.pow(3);
        let rows = |f: &mut dyn FnMut(usize, usize)| {
            for k in 0..wz {
                for j in 0..wy {
                    f((k * wy + j) * wx, w.lo[0] + nx * ((w.lo[1] + j) + ny * (w.lo[2] + k)));
                }
            }
        };
        for (dst, src) in self.vox.f.iter_mut().zip(&part.f) {
            if !dst.is_empty() {
                rows(&mut |s, d| dst[d..d + wx].copy_from_slice(&src[s..s + wx]));
            }
        }
        for (dst, src) in self.vox.i.iter_mut().zip(&part.i) {
            rows(&mut |s, d| dst[d..d + wx].copy_from_slice(&src[s..s + wx]));
        }
        if let (Some(dst), Some(src)) = (self.vox.ppl_lambda.as_mut(), part.ppl_lambda.as_ref()) {
            rows(&mut |s, d| dst[d..d + wx].copy_from_slice(&src[s..s + wx]));
        }
        if let (Some(dst), Some(src)) = (self.vox.subvoxel_counts.as_mut(), part.subvoxel_counts.as_ref()) {
            rows(&mut |s, d| dst[d * n_sub..(d + wx) * n_sub].copy_from_slice(&src[s * n_sub..(s + wx) * n_sub]));
        }
    }
}

/// An empty grid of `shape` with the fields `params` switches on.
pub(crate) fn empty_grid(params: &VoxelParams, origin: crate::Point, shape: [usize; 3]) -> RayVoxels {
    let n = shape[0] * shape[1] * shape[2];
    let ppl = params.attenuation.contains(&Attenuation::Ppl);
    let n_sub = params.subvoxel_split.pow(3);
    RayVoxels {
        origin,
        voxel_size: params.voxel_size,
        shape,
        params: params.clone(),
        has_leaf: false,
        has_wood: false,
        f: super::F::ALL
            .iter()
            .map(|fld| {
                let on = if fld.is_beam() { params.beam.is_some() } else if *fld == super::F::PplMissWl { ppl } else { true };
                if on { vec![0.0f32; n] } else { Vec::new() }
            })
            .collect(),
        i: super::I::ALL.iter().map(|_| vec![0i32; n]).collect(),
        ppl_lambda: ppl.then(|| vec![-1.0f32; n]),
        subvoxel_counts: (n_sub > 0).then(|| vec![0u8; n * n_sub]),
        ground_height: None,
        wood_volume: None,
        predominant_tree: None,
        tree_iad: Default::default(),
    }
}

/// Trace `pulses` through a grid block by block (see the module docs).
///
/// With `out` the blocks are written to that directory as they finish (see
/// [`BlockedGrid`]) and `None` is returned; neighbour priors and inclination
/// distributions, which need the whole grid, are then refused. Without it
/// the blocks are assembled into one [`RayVoxels`], which must fit in memory
/// (about 0.13 kB per voxel instead of the 0.35 kB a whole-grid trace takes
/// at its peak).
pub fn voxelize_blocks(pulses: &Pulses, params: &VoxelParams, dtm: Option<&Raster>, opts: &BlockOptions, out: Option<&Path>) -> Result<(Option<RayVoxels>, BlockStats)> {
    validate(params, opts)?;
    let wants_iad = params.inclination || params.attenuation.contains(&Attenuation::Bailey);
    if out.is_some() && (wants_iad || params.neighbour_prior_min_rays > 0) {
        return Err(Error::invalid("neighbour priors, inclination distributions and the bailey method need the whole grid; trace without an output directory"));
    }
    let bounds = match (params.bounds, pulses) {
        (Some(b), _) => b,
        (None, Pulses::File { file, .. }) => pad_bounds(file.bounds),
        (None, Pulses::Memory(inputs)) => {
            let xyz = inputs.shots.echo_xyz();
            if xyz.is_empty() {
                return Err(Error::invalid("no echoes to fit the grid to; pass bounds"));
            }
            pad_bounds((crate::util::spatial::min_corner(&xyz), crate::util::spatial::max_corner(&xyz)))
        }
    };
    let shape = grid_shape(params, bounds)?;
    let lo = bounds.0;
    let geom = Geom::new(lo, params.voxel_size, shape);
    let ground_height = dtm.map(|d| column_ground(d, lo, params.voxel_size, shape));
    let block: [usize; 3] = std::array::from_fn(|k| opts.block[k].min(shape[k]));
    let nb: [usize; 3] = std::array::from_fn(|k| shape[k].div_ceil(block[k]));
    let per_voxel = bytes_per_voxel(params);
    let allowance = opts.max_memory.unwrap_or_else(|| crate::util::limits::budget().map_or(8_000_000_000, |b| b / 2));
    let budget = (allowance / per_voxel as u64).max(1) as usize;
    let passes = plan_passes(shape, block, nb, budget)?;

    let assembly = match out {
        Some(_) => None,
        None => {
            let n = shape.iter().map(|&v| v as u128).product::<u128>();
            let fields = super::F::ALL.len() + super::I::ALL.len() + 1;
            crate::util::limits::check_cells(
                n,
                (fields * 4 + params.subvoxel_split.pow(3)) as u64,
                &format!("the assembled {} x {} x {} voxel grid at {} m", shape[0], shape[1], shape[2], params.voxel_size),
                "an output directory, so that blocks are written as they finish",
            )?;
            Some(Assembly { vox: empty_grid(params, lo, shape) })
        }
    };
    let writer = match out {
        Some(dir) => Some(BlockWriter::create(dir, &BlockedGrid::new(dir, lo, shape, block, params.clone(), ground_height.clone()))?),
        None => None,
    };

    let pool = if opts.workers > 0 { Some(rayon::ThreadPoolBuilder::new().num_threads(opts.workers).build().map_err(|e| Error::invalid(format!("thread pool: {e}")))?) } else { None };
    let run = move || -> Result<(Option<RayVoxels>, BlockStats)> {
        let (mut assembly, writer) = (assembly, writer);
        let mut stats = BlockStats { n_blocks: nb, n_passes: passes.len(), bytes_per_voxel: per_voxel, ..Default::default() };
        // Column peaks for flat_top, before any pulse is traced.
        let peaks = if params.flat_top {
            let mut p = vec![f64::MIN; shape[0] * shape[1]];
            for_each_batch(pulses, dtm, |inputs, range| {
                let shots = inputs.shots;
                let xyz: Vec<crate::Point> = range.flat_map(|s| (shots.echo_start[s]..shots.echo_start[s] + shots.echo_count[s] as usize).map(move |e| crate::transform::add(&shots.origin[s], &crate::transform::scale(&shots.direction[s], shots.echo_range[e])))).collect();
                refine::update_peaks(&mut p, &xyz, &geom);
                Ok(())
            })?;
            Some(p)
        } else {
            None
        };
        let (mut has_leaf, mut has_wood) = (false, false);
        let mut echoes = (wants_iad && assembly.is_some()).then(iad::EchoPoints::default);
        let total: u64 = match pulses {
            Pulses::Memory(i) => i.shots.n_shots() as u64,
            Pulses::File { file, .. } => file.n_shots as u64,
        };
        for (p, pass) in passes.iter().enumerate() {
            let mut engines: Vec<Engine> = pass.windows.par_iter().map(|w| Engine::new_window(params, geom.clone(), *w)).collect();
            stats.peak_voxels = stats.peak_voxels.max(engines.iter().map(|e| e.n_cells()).sum());
            let task = crate::util::progress::start(format!("tracing blocks, pass {} of {}", p + 1, passes.len()), total);
            for_each_batch(pulses, dtm, |inputs, range| {
                // A batch starting at pulse 0 is a new set of inputs: all of
                // them in memory, or the next row groups of a file.
                if p == 0 && range.start == 0 {
                    check_inputs(inputs, params)?;
                    has_leaf |= inputs.foliage.is_some_and(|f| f.contains(&foliage::LEAF));
                    has_wood |= inputs.foliage.is_some_and(|f| f.contains(&foliage::WOOD));
                    if let Some(e) = &mut echoes {
                        e.collect(inputs, &geom);
                    }
                }
                let lists = bin_pulses(inputs, range.clone(), &geom, params, block, nb, pass);
                stats.block_pulses += lists.iter().map(|l| l.len() as u64).sum::<u64>();
                let base = range.start as u32;
                engines.par_iter_mut().zip(lists.par_iter()).for_each(|(e, list)| {
                    let mut scratch = Scratch::default();
                    let ids: Vec<u32> = list.iter().map(|&s| s + base).collect();
                    e.add_sequential(inputs, &ids, peaks.as_deref(), ground_height.as_deref(), &mut scratch);
                });
                task.inc(range.len() as u64);
                Ok(())
            })?;
            drop(task);
            let asm = Mutex::new(assembly.as_mut());
            let written: Vec<bool> = engines
                .into_par_iter()
                .zip(pass.windows.par_iter())
                .map(|(e, w)| -> Result<bool> {
                    let part = e.finish();
                    match &writer {
                        Some(wr) => wr.write_block(w, &part),
                        None => {
                            if let Some(a) = asm.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                                a.put(w, &part);
                            }
                            Ok(false)
                        }
                    }
                })
                .collect::<Result<_>>()?;
            stats.blocks_written += written.iter().filter(|&&b| b).count();
        }
        if params.attenuation.contains(&Attenuation::Bailey) && !(has_leaf && has_wood) {
            return Err(Error::invalid("the bailey method needs both leaf and wood echoes"));
        }
        if let Some(w) = &writer {
            w.finish(has_leaf, has_wood)?;
        }
        let grid = match assembly.take() {
            None => None,
            Some(Assembly { mut vox }) => {
                vox.has_leaf = has_leaf;
                vox.has_wood = has_wood;
                vox.ground_height = ground_height.clone();
                if params.neighbour_prior_min_rays > 0 {
                    refine::apply_neighbour_priors(&mut vox, params.neighbour_prior_min_rays as f32);
                }
                if let Some(e) = echoes.take() {
                    iad::build(&mut vox, e);
                }
                Some(vox)
            }
        };
        Ok((grid, stats))
    };
    match &pool {
        Some(p) => p.install(run),
        None => run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every block a segment passes within the margin of, by brute force.
    fn brute(a: &[f64; 3], b: &[f64; 3], block: [usize; 3], nb: [usize; 3], m: f64) -> Vec<[usize; 3]> {
        let mut out = Vec::new();
        for bz in 0..nb[2] {
            for by in 0..nb[1] {
                for bx in 0..nb[0] {
                    let w = Window { lo: [bx * block[0], by * block[1], bz * block[2]], shape: block };
                    if w.touches(a, b, m) {
                        out.push([bx, by, bz]);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn segment_blocks_match_a_slab_test_of_every_block() {
        let mut seed = 7u64;
        let mut rnd = |lo: f64, hi: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((seed >> 11) as f64 / (1u64 << 53) as f64)
        };
        let (block, nb) = ([4, 5, 3], [6, 5, 7]);
        for n in 0..2000 {
            let mut a = [rnd(-5.0, 30.0), rnd(-5.0, 30.0), rnd(-5.0, 25.0)];
            let mut b = [rnd(-5.0, 30.0), rnd(-5.0, 30.0), rnd(-5.0, 25.0)];
            // Axis-parallel segments and segments along block faces and edges.
            match n % 5 {
                1 => b[1] = a[1],
                2 => {
                    a[0] = 8.0;
                    b[0] = 8.0;
                }
                3 => {
                    a = [4.0, 10.0, rnd(-2.0, 22.0)];
                    b = [4.0, 10.0, rnd(-2.0, 22.0)];
                }
                _ => {}
            }
            let mut got = Vec::new();
            blocks_on_segment(&a, &b, block, [0; 3], nb, MARGIN, &mut got);
            got.sort_by_key(|v| (v[2], v[1], v[0]));
            // Never a block the segment reaches missed; nothing beyond twice the margin.
            let (inner, outer) = (brute(&a, &b, block, nb, MARGIN), brute(&a, &b, block, nb, 2.0 * MARGIN));
            assert!(inner.iter().all(|v| got.contains(v)), "{a:?} {b:?}");
            assert!(got.iter().all(|v| outer.contains(v)), "{a:?} {b:?}");
        }
    }

    #[test]
    fn passes_hold_at_most_the_budget() {
        let shape = [10, 7, 9];
        let block = [4, 4, 4];
        let nb = [3, 2, 3];
        let passes = plan_passes(shape, block, nb, 150).unwrap();
        let total: usize = passes.iter().flat_map(|p| p.windows.iter().map(|w| w.n_voxels())).sum();
        assert_eq!(total, 10 * 7 * 9);
        assert!(passes.iter().all(|p| p.windows.iter().map(|w| w.n_voxels()).sum::<usize>() <= 150));
        assert!(plan_passes(shape, block, nb, 63).is_err());
    }
}
