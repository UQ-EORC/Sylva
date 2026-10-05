// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A ray-traced voxel grid kept on disk block by block.
//!
//! A directory holds `grid.json` (grid placement, block size, tracing
//! options, echo classes seen), `ground_height.f64` (terrain height under
//! each column, little-endian, `i + nx j`, when a DTM was given) and one
//! Parquet file per block, `block_<bx>_<by>_<bz>.parquet`, with one row per
//! voxel of the block (x fastest, then y, then z) and one column per raw
//! accumulator: the single-precision sums, the integer counts,
//! `ppl_lambda` and `subvoxel_counts` when traced. Blocks that no pulse
//! reached are not written and read as zeros.
//!
//! Any box of the grid can be read back as a [`RayVoxels`], so every derived
//! quantity works on a block or a slab; the layer summaries, the `.vox`
//! writer and the per-tree sampling run over the grid one slab of blocks at
//! a time.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::column::reader::ColumnReader;
use parquet::data_type::{ByteArray, FixedLenByteArray, FixedLenByteArrayType, FloatType, Int32Type};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use rayon::prelude::*;

use super::blocks::empty_grid;
use super::quality::{tree_sampling_with, TreeSampling};
use super::traverse::Window;
use super::write::WriteOptions;
use super::{Attenuation, BeamSpec, Lad, RayVoxels, VoxelParams, WeightMethod, F, I};
use crate::error::{Error, Result};
use crate::util::json::{self, Json};
use crate::util::numeric::{arange, searchsorted_right};
use crate::voxel::grid::{FieldData, OcclusionProfile, EMPTY, FILLED, OCCLUDED};
use crate::Point;

const FORMAT: &str = "sylva-blocked-voxels";
const VERSION: i64 = 1;

/// A voxel grid stored as blocks in a directory (see the module docs).
#[derive(Debug, Clone)]
pub struct BlockedGrid {
    pub dir: PathBuf,
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// Voxels per block along x, y and z.
    pub block: [usize; 3],
    pub params: VoxelParams,
    pub has_leaf: bool,
    pub has_wood: bool,
    /// Terrain height under each column (`i + nx j`).
    pub ground_height: Option<Vec<f64>>,
    /// Blocks on disk.
    present: BTreeSet<[usize; 3]>,
}

fn weighting_name(w: WeightMethod) -> &'static str {
    match w {
        WeightMethod::Equal => "equal",
        WeightMethod::Full => "full",
        WeightMethod::First => "first",
        WeightMethod::Relative => "relative",
        WeightMethod::Strongest => "strongest",
    }
}

fn lad_params(l: &Lad) -> Vec<f64> {
    match *l {
        Lad::Ellipsoidal(chi) => vec![chi],
        Lad::TwoParamBeta(mu, nu) => vec![mu, nu],
        _ => vec![],
    }
}

fn num(x: f64) -> Json {
    if x.is_finite() { Json::Float(x) } else { Json::Null }
}

fn arr(v: impl IntoIterator<Item = Json>) -> Json {
    Json::Array(v.into_iter().collect())
}

fn obj(items: Vec<(&str, Json)>) -> Json {
    Json::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn block_name(b: [usize; 3]) -> String {
    format!("block_{}_{}_{}.parquet", b[0], b[1], b[2])
}

impl BlockedGrid {
    pub(crate) fn new(dir: &Path, origin: Point, shape: [usize; 3], block: [usize; 3], params: VoxelParams, ground_height: Option<Vec<f64>>) -> Self {
        BlockedGrid { dir: dir.to_path_buf(), origin, voxel_size: params.voxel_size, shape, block, params, has_leaf: false, has_wood: false, ground_height, present: BTreeSet::new() }
    }

    /// Blocks along x, y and z.
    pub fn n_blocks(&self) -> [usize; 3] {
        std::array::from_fn(|k| self.shape[k].div_ceil(self.block[k]))
    }

    pub fn n_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Blocks written to disk.
    pub fn blocks_present(&self) -> Vec<[usize; 3]> {
        self.present.iter().copied().collect()
    }

    fn manifest(&self, complete: bool) -> Json {
        let p = &self.params;
        let ints = |v: &[usize]| arr(v.iter().map(|&x| Json::Int(x as i64)));
        obj(vec![
            ("format", Json::Str(FORMAT.into())),
            ("version", Json::Int(VERSION)),
            ("complete", Json::Bool(complete)),
            ("origin", arr(self.origin.iter().map(|&x| Json::Float(x)))),
            ("voxel_size", Json::Float(self.voxel_size)),
            ("shape", ints(&self.shape)),
            ("block", ints(&self.block)),
            ("has_leaf", Json::Bool(self.has_leaf)),
            ("has_wood", Json::Bool(self.has_wood)),
            ("ground_height", Json::Bool(self.ground_height.is_some())),
            ("blocks", arr(self.present.iter().map(|b| ints(b)))),
            (
                "params",
                obj(vec![
                    ("weighting", Json::Str(weighting_name(p.weighting).into())),
                    ("occlusion", Json::Bool(p.occlusion)),
                    ("flat_top", Json::Bool(p.flat_top)),
                    ("neighbour_prior_min_rays", Json::Int(p.neighbour_prior_min_rays as i64)),
                    ("beam", p.beam.map_or(Json::Null, |b| arr([Json::Float(b.diameter), Json::Float(b.divergence)]))),
                    ("subvoxel_split", Json::Int(p.subvoxel_split as i64)),
                    ("subvoxel_min_beams", Json::Int(p.subvoxel_min_beams as i64)),
                    ("average_leaf_area", Json::Float(p.average_leaf_area)),
                    ("lad", Json::Str(p.lad.name().into())),
                    ("lad_params", arr(lad_params(&p.lad).into_iter().map(Json::Float))),
                    ("attenuation", arr(p.attenuation.iter().map(|m| Json::Str(m.name().into())))),
                    ("inclination", Json::Bool(p.inclination)),
                    ("n_iad_bins", Json::Int(p.n_iad_bins as i64)),
                    ("knn_normal", Json::Int(p.knn_normal as i64)),
                    ("triangle_lmax", Json::Float(p.triangle_lmax)),
                    ("unbounded_range", num(p.unbounded_range)),
                ]),
            ),
        ])
    }

    fn save_manifest(&self, complete: bool) -> Result<()> {
        let path = self.dir.join("grid.json");
        let tmp = self.dir.join("grid.json.tmp");
        std::fs::write(&tmp, json::to_string_indented(&self.manifest(complete), 2) + "\n").map_err(|e| Error::file(&tmp, e.to_string()))?;
        std::fs::rename(&tmp, &path).map_err(|e| Error::file(&path, e.to_string()))
    }

    /// Open a blocked grid written by [`super::voxelize_blocks`].
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("grid.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::file(&path, e.to_string()))?;
        let bad = |what: &str| Error::file(&path, format!("not a blocked voxel grid ({what})"));
        let j = json::parse(&text)?;
        if j.get("format") != Some(&Json::Str(FORMAT.into())) {
            return Err(bad("format"));
        }
        if j.get("version") != Some(&Json::Int(VERSION)) {
            return Err(bad("unknown version"));
        }
        if j.get("complete") != Some(&Json::Bool(true)) {
            return Err(Error::file(&path, "the blocked grid is incomplete: its trace did not finish"));
        }
        let f = |v: Option<&Json>| -> Option<f64> {
            match v? {
                Json::Float(x) => Some(*x),
                Json::Int(i) => Some(*i as f64),
                Json::Null => Some(f64::INFINITY),
                _ => None,
            }
        };
        let u = |v: Option<&Json>| -> Option<usize> {
            match v? {
                Json::Int(i) if *i >= 0 => Some(*i as usize),
                _ => None,
            }
        };
        let b = |v: Option<&Json>| -> Option<bool> {
            match v? {
                Json::Bool(x) => Some(*x),
                _ => None,
            }
        };
        let list = |v: Option<&Json>| -> Option<Vec<Json>> {
            match v? {
                Json::Array(a) => Some(a.clone()),
                _ => None,
            }
        };
        let triple_f = |v: Option<&Json>| -> Option<Point> {
            let a = list(v)?;
            (a.len() == 3).then(|| Some([f(a.first())?, f(a.get(1))?, f(a.get(2))?]))?
        };
        let triple_u = |v: &Json| -> Option<[usize; 3]> {
            let Json::Array(a) = v else { return None };
            (a.len() == 3).then(|| Some([u(a.first())?, u(a.get(1))?, u(a.get(2))?]))?
        };
        let origin = triple_f(j.get("origin")).ok_or_else(|| bad("origin"))?;
        let voxel_size = f(j.get("voxel_size")).ok_or_else(|| bad("voxel_size"))?;
        let shape = j.get("shape").and_then(triple_u).ok_or_else(|| bad("shape"))?;
        let block = j.get("block").and_then(triple_u).ok_or_else(|| bad("block"))?;
        let p = j.get("params").ok_or_else(|| bad("params"))?;
        let s = |k: &str| -> Result<String> {
            match p.get(k) {
                Some(Json::Str(v)) => Ok(v.clone()),
                _ => Err(bad(k)),
            }
        };
        let lad_p: Vec<f64> = list(p.get("lad_params")).ok_or_else(|| bad("lad_params"))?.iter().map(|v| f(Some(v))).collect::<Option<_>>().ok_or_else(|| bad("lad_params"))?;
        let beam = match p.get("beam") {
            Some(Json::Null) | None => None,
            Some(v) => {
                let a = list(Some(v)).ok_or_else(|| bad("beam"))?;
                Some(BeamSpec { diameter: f(a.first()).ok_or_else(|| bad("beam"))?, divergence: f(a.get(1)).ok_or_else(|| bad("beam"))? })
            }
        };
        let params = VoxelParams {
            voxel_size,
            bounds: Some((origin, std::array::from_fn(|k| origin[k] + shape[k] as f64 * voxel_size))),
            weighting: WeightMethod::parse(&s("weighting")?)?,
            occlusion: b(p.get("occlusion")).ok_or_else(|| bad("occlusion"))?,
            flat_top: b(p.get("flat_top")).ok_or_else(|| bad("flat_top"))?,
            neighbour_prior_min_rays: u(p.get("neighbour_prior_min_rays")).ok_or_else(|| bad("neighbour_prior_min_rays"))? as u32,
            beam,
            subvoxel_split: u(p.get("subvoxel_split")).ok_or_else(|| bad("subvoxel_split"))?,
            subvoxel_min_beams: u(p.get("subvoxel_min_beams")).ok_or_else(|| bad("subvoxel_min_beams"))? as u8,
            average_leaf_area: f(p.get("average_leaf_area")).ok_or_else(|| bad("average_leaf_area"))?,
            lad: Lad::parse(&s("lad")?, &lad_p)?,
            attenuation: list(p.get("attenuation")).ok_or_else(|| bad("attenuation"))?.iter().map(|m| match m {
                Json::Str(m) => Attenuation::parse(m),
                _ => Err(bad("attenuation")),
            }).collect::<Result<_>>()?,
            inclination: b(p.get("inclination")).ok_or_else(|| bad("inclination"))?,
            n_iad_bins: u(p.get("n_iad_bins")).ok_or_else(|| bad("n_iad_bins"))?,
            knn_normal: u(p.get("knn_normal")).ok_or_else(|| bad("knn_normal"))?,
            triangle_lmax: f(p.get("triangle_lmax")).ok_or_else(|| bad("triangle_lmax"))?,
            unbounded_range: f(p.get("unbounded_range")).ok_or_else(|| bad("unbounded_range"))?,
        };
        let ground_height = if b(j.get("ground_height")).ok_or_else(|| bad("ground_height"))? {
            let gp = dir.join("ground_height.f64");
            let bytes = std::fs::read(&gp).map_err(|e| Error::file(&gp, e.to_string()))?;
            if bytes.len() != 8 * shape[0] * shape[1] {
                return Err(Error::file(&gp, "wrong size for the grid"));
            }
            Some(bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect())
        } else {
            None
        };
        let present = list(j.get("blocks")).ok_or_else(|| bad("blocks"))?.iter().map(triple_u).collect::<Option<_>>().ok_or_else(|| bad("blocks"))?;
        Ok(BlockedGrid {
            dir: dir.to_path_buf(),
            origin,
            voxel_size,
            shape,
            block,
            params,
            has_leaf: b(j.get("has_leaf")).ok_or_else(|| bad("has_leaf"))?,
            has_wood: b(j.get("has_wood")).ok_or_else(|| bad("has_wood"))?,
            ground_height,
            present,
        })
    }

    fn window_of(&self, b: [usize; 3]) -> Window {
        let lo: [usize; 3] = std::array::from_fn(|k| b[k] * self.block[k]);
        Window { lo, shape: std::array::from_fn(|k| self.block[k].min(self.shape[k] - lo[k])) }
    }

    /// Voxels `lo..hi` (indices, `hi` exclusive) as a grid of their own,
    /// with its origin at voxel `lo`.
    pub fn read_box(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<RayVoxels> {
        for k in 0..3 {
            if lo[k] >= hi[k] || hi[k] > self.shape[k] {
                return Err(Error::invalid(format!("voxel box {lo:?}..{hi:?} is empty or outside the {:?} grid", self.shape)));
            }
        }
        let shape: [usize; 3] = std::array::from_fn(|k| hi[k] - lo[k]);
        crate::util::limits::check_cells(shape.iter().map(|&v| v as u128).product(), 4 * (F::COUNT + I::COUNT + 1) as u64 + self.params.subvoxel_split.pow(3) as u64, &format!("a {} x {} x {} voxel box", shape[0], shape[1], shape[2]), "a smaller box")?;
        let origin: Point = std::array::from_fn(|k| self.origin[k] + lo[k] as f64 * self.voxel_size);
        let mut out = empty_grid(&self.params, origin, shape);
        out.has_leaf = self.has_leaf;
        out.has_wood = self.has_wood;
        out.ground_height = self.ground_height.as_ref().map(|g| (lo[1]..hi[1]).flat_map(|j| (lo[0]..hi[0]).map(move |i| g[i + self.shape[0] * j])).collect());
        let blocks: Vec<[usize; 3]> = self.present.iter().copied().filter(|b| (0..3).all(|k| b[k] * self.block[k] < hi[k] && (b[k] + 1) * self.block[k] > lo[k])).collect();
        let n_sub = self.params.subvoxel_split.pow(3);
        let dst = Mutex::new(&mut out);
        blocks.par_iter().try_for_each(|&b| -> Result<()> {
            let w = self.window_of(b);
            let part = read_block(&self.dir.join(block_name(b)), &self.params, w.shape)?;
            // The overlap, in grid indices.
            let a: [usize; 3] = std::array::from_fn(|k| w.lo[k].max(lo[k]));
            let z: [usize; 3] = std::array::from_fn(|k| (w.lo[k] + w.shape[k]).min(hi[k]));
            let len = z[0] - a[0];
            let mut g = dst.lock().unwrap_or_else(|e| e.into_inner());
            for k in a[2]..z[2] {
                for j in a[1]..z[1] {
                    let s = (a[0] - w.lo[0]) + w.shape[0] * ((j - w.lo[1]) + w.shape[1] * (k - w.lo[2]));
                    let d = (a[0] - lo[0]) + shape[0] * ((j - lo[1]) + shape[1] * (k - lo[2]));
                    for (dv, sv) in g.f.iter_mut().zip(&part.f) {
                        if !dv.is_empty() {
                            dv[d..d + len].copy_from_slice(&sv[s..s + len]);
                        }
                    }
                    for (dv, sv) in g.i.iter_mut().zip(&part.i) {
                        dv[d..d + len].copy_from_slice(&sv[s..s + len]);
                    }
                    if let (Some(dv), Some(sv)) = (g.ppl_lambda.as_mut(), part.ppl_lambda.as_ref()) {
                        dv[d..d + len].copy_from_slice(&sv[s..s + len]);
                    }
                    if let (Some(dv), Some(sv)) = (g.subvoxel_counts.as_mut(), part.subvoxel_counts.as_ref()) {
                        dv[d * n_sub..(d + len) * n_sub].copy_from_slice(&sv[s * n_sub..(s + len) * n_sub]);
                    }
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// Block `[bx, by, bz]` as a grid of its own.
    pub fn block(&self, b: [usize; 3]) -> Result<RayVoxels> {
        let nb = self.n_blocks();
        if (0..3).any(|k| b[k] >= nb[k]) {
            return Err(Error::invalid(format!("block {b:?} is outside the {nb:?} blocks")));
        }
        let w = self.window_of(b);
        self.read_box(w.lo, std::array::from_fn(|k| w.lo[k] + w.shape[k]))
    }

    /// The whole grid in memory.
    pub fn to_grid(&self) -> Result<RayVoxels> {
        let mut g = self.read_box([0; 3], self.shape)?;
        g.origin = self.origin;
        Ok(g)
    }

    /// Voxel layers `k0..k1` a slab of blocks at a time, bottom first.
    ///
    /// Returning `impl Iterator` hands back a lazy sequence rather than a list:
    /// the slab bounds are worked out as the caller asks for them, so a grid of
    /// any height is walked without holding more than one slab in memory. The
    /// `+ '_` says the sequence borrows `self` and so cannot outlive the grid.
    fn slabs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.n_blocks()[2]).map(|bz| (bz * self.block[2], ((bz + 1) * self.block[2]).min(self.shape[2])))
    }

    fn read_slab(&self, k0: usize, k1: usize) -> Result<RayVoxels> {
        self.read_box([0, 0, k0], [self.shape[0], self.shape[1], k1])
    }

    /// Names of the raw accumulators.
    pub fn field_names(&self) -> Vec<&'static str> {
        empty_grid(&self.params, self.origin, [1, 1, 1]).field_names().into_iter().chain(self.ground_height.is_some().then_some("ground_height")).collect()
    }

    /// A raw accumulator of the whole grid, assembled slab by slab.
    pub fn field(&self, name: &str) -> Result<FieldData> {
        if name == "ground_height" {
            return self.ground_height.clone().map(FieldData::F64).ok_or_else(|| Error::invalid("no voxel field \"ground_height\" (see field_names())"));
        }
        let per = self.shape[0] * self.shape[1];
        let mut out: Option<FieldData> = None;
        for (k0, k1) in self.slabs() {
            let part = self.read_slab(k0, k1)?.field(name)?;
            let at = k0 * per * if name == "subvoxel_counts" { self.params.subvoxel_split.pow(3) } else { 1 };
            let n = self.n_voxels() * if name == "subvoxel_counts" { self.params.subvoxel_split.pow(3) } else { 1 };
            macro_rules! put {
                ($variant:ident, $v:expr, $zero:expr) => {{
                    let o = out.get_or_insert_with(|| FieldData::$variant(vec![$zero; n]));
                    if let FieldData::$variant(o) = o {
                        o[at..at + $v.len()].copy_from_slice(&$v);
                    }
                }};
            }
            match part {
                FieldData::F32(v) => put!(F32, v, 0.0f32),
                FieldData::I32(v) => put!(I32, v, 0i32),
                FieldData::F64(v) => put!(F64, v, 0.0f64),
                FieldData::U8(v) => put!(U8, v, 0u8),
            }
        }
        out.ok_or_else(|| Error::invalid("empty grid"))
    }

    /// A derived quantity of the whole grid (see [`RayVoxels::metric`]).
    /// Voxel centres, and so `distance_from_ground`, are computed from each
    /// slab's own origin: they can differ from a whole grid's in the last bit.
    pub fn metric(&self, name: &str) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(self.n_voxels());
        for (k0, k1) in self.slabs() {
            out.extend(self.read_slab(k0, k1)?.metric(name)?);
        }
        Ok(out)
    }

    /// Mean of a field or metric per layer (see [`RayVoxels::profile`]); a
    /// slab holds whole layers, so this is the whole grid's profile.
    pub fn profile(&self, name: &str, min_beams: f64) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(self.shape[2]);
        for (k0, k1) in self.slabs() {
            out.extend(self.read_slab(k0, k1)?.profile(name, min_beams)?);
        }
        Ok(out)
    }

    /// Visit every voxel's height (above the ground where the DTM has any
    /// finite height, else above the grid floor) and state, a slab at a time,
    /// in flat index order; `f(idx, height, state, beams)`.
    fn each_voxel(&self, mut f: impl FnMut(usize, f64, u8, i32)) -> Result<()> {
        let per = self.shape[0] * self.shape[1];
        let ground = self.ground_height.as_ref().filter(|g| g.iter().any(|v| v.is_finite()));
        for (k0, k1) in self.slabs() {
            let slab = self.read_slab(k0, k1)?;
            for local in 0..slab.n_voxels() {
                let idx = k0 * per + local;
                let k = idx / per;
                let h = match ground {
                    // The whole grid's centre, from the grid origin.
                    Some(g) => (self.origin[2] + (k as f64 + 0.5) * self.voxel_size) - g[idx % per],
                    None => (k as f64 + 0.5) * self.voxel_size,
                };
                f(idx, h, slab.state(local) as u8, slab.get_i(I::NumBeams, local));
            }
        }
        Ok(())
    }

    fn canopy_top(&self, max_height: Option<f64>) -> Result<f64> {
        if let Some(t) = max_height {
            return Ok(t);
        }
        let (mut any, mut top) = (false, f64::NEG_INFINITY);
        self.each_voxel(|_, h, s, _| {
            if s == FILLED {
                any = true;
                if !h.is_nan() {
                    top = top.max(h);
                }
            }
        })?;
        Ok(if !any { 0.0 } else if top == f64::NEG_INFINITY { f64::NAN } else { top })
    }

    /// What the scan saw of the canopy space (see
    /// [`RayVoxels::occlusion_profile`]), read a slab at a time.
    pub fn occlusion_profile(&self, min_height: f64, max_height: Option<f64>) -> Result<OcclusionProfile> {
        let top = self.canopy_top(max_height)?;
        let vs = self.voxel_size;
        if !(top + vs - min_height).is_finite() {
            return Err(Error::invalid("arange: cannot compute length"));
        }
        let edges = arange(min_height, top + vs, vs);
        let n_layers = edges.len().saturating_sub(1).max(1);
        let kmax = edges.len().saturating_sub(2);
        let mut n = vec![0i64; n_layers];
        let mut obs = vec![0i64; n_layers];
        let mut occ = vec![0i64; n_layers];
        let mut bm = vec![0.0f64; n_layers];
        self.each_voxel(|_, h, s, beams| {
            if !(h.is_finite() && h >= min_height && h <= top) {
                return;
            }
            let k = (searchsorted_right(&edges, h) as i64 - 1).clamp(0, kmax as i64) as usize;
            if k >= n_layers {
                return;
            }
            n[k] += 1;
            if s >= EMPTY {
                obs[k] += 1;
            }
            if s == OCCLUDED {
                occ[k] += 1;
            }
            bm[k] += beams as f64;
        })?;
        let height: Vec<f64> = edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).take(n_layers).collect();
        let div = |a: f64, b: i64| a / b as f64;
        let tot = n.iter().sum::<i64>().max(1) as f64;
        let (sn, so, sc) = (n.iter().sum::<i64>(), obs.iter().sum::<i64>(), occ.iter().sum::<i64>());
        Ok(OcclusionProfile {
            height,
            observed: obs.iter().zip(&n).map(|(&o, &c)| div(o as f64, c)).collect(),
            occluded: occ.iter().zip(&n).map(|(&o, &c)| div(o as f64, c)).collect(),
            unobserved: (0..n_layers).map(|k| div((n[k] - obs[k] - occ[k]) as f64, n[k])).collect(),
            mean_beams: bm.iter().zip(&n).map(|(&b, &c)| div(b, c)).collect(),
            n_voxels: n,
            total_observed: so as f64 / tot,
            total_occluded: sc as f64 / tot,
            total_unobserved: (sn - so - sc) as f64 / tot,
            top,
        })
    }

    /// Share of each column's canopy space that was observed, `(ny, nx)`
    /// (see [`RayVoxels::observed_map`]).
    pub fn observed_map(&self, min_height: f64, max_height: Option<f64>) -> Result<Vec<f64>> {
        let top = self.canopy_top(max_height)?;
        let per = self.shape[0] * self.shape[1];
        let mut seen = vec![0i64; per];
        let mut all = vec![0i64; per];
        self.each_voxel(|idx, h, s, _| {
            if h.is_finite() && h >= min_height && h <= top {
                all[idx % per] += 1;
                if s >= EMPTY {
                    seen[idx % per] += 1;
                }
            }
        })?;
        Ok(seen.iter().zip(&all).map(|(&s, &a)| s as f64 / a as f64).collect())
    }

    /// How well each tree was seen (see [`super::quality::tree_sampling`]).
    /// Holds the state and pulse count of every voxel (5 bytes each).
    pub fn tree_sampling(&self, points: &[Point], labels: &[i64], min_beams: f64, above: f64) -> Result<Vec<TreeSampling>> {
        if labels.len() != points.len() {
            return Err(Error::invalid("labels must match the points, state and beams the grid"));
        }
        let n = self.n_voxels();
        crate::util::limits::check_cells(n as u128, 5, "voxel states and pulse counts for the tree sampling", "a coarser grid")?;
        let mut state = vec![0u8; n];
        let mut beams = vec![0i32; n];
        self.each_voxel(|idx, _, s, b| {
            state[idx] = s;
            beams[idx] = b;
        })?;
        Ok(tree_sampling_with(points, labels, self.origin, self.voxel_size, self.shape, &state, &|i| beams[i] as f64, min_beams, above))
    }

    /// Write an AMAPVox `.vox` file (`text = false`) or a table with voxel
    /// centres, a slab at a time; the same rows as the whole grid's writer.
    pub fn write(&self, path: impl AsRef<Path>, text: bool, opts: WriteOptions) -> Result<usize> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|e| Error::file(path, e.to_string()))?;
        let mut out = BufWriter::new(file);
        let mut written = 0;
        for (n, (k0, k1)) in self.slabs().enumerate() {
            let slab = self.read_slab(k0, k1)?;
            let cols = slab.columns();
            if n == 0 {
                if text {
                    slab.text_header(&mut out, &cols)?;
                } else {
                    slab.vox_header(&mut out, &cols, self.origin, self.shape)?;
                }
            }
            written += slab.write_rows(&mut out, &cols, opts, text, [0, 0, k0])?;
        }
        out.flush()?;
        Ok(written)
    }
}

/// Writes the blocks of a trace as they finish.
///
/// Blocks are traced in parallel and finish in no particular order, so the
/// manifest they all update sits behind a `Mutex`: a thread takes the lock,
/// records its block, and releases it. Only the bookkeeping is serialised —
/// each block's Parquet file is written by its own thread, outside the lock.
pub(crate) struct BlockWriter {
    grid: Mutex<BlockedGrid>,
}

impl BlockWriter {
    /// Start a blocked grid in `dir` (created if needed; a grid already
    /// there is replaced).
    pub fn create(dir: &Path, grid: &BlockedGrid) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| Error::file(dir, e.to_string()))?;
        // Clear what an earlier grid left.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if (name.starts_with("block_") && name.ends_with(".parquet")) || name == "ground_height.f64" || name == "grid.json" {
                    std::fs::remove_file(e.path()).map_err(|err| Error::file(e.path(), err.to_string()))?;
                }
            }
        }
        if let Some(g) = &grid.ground_height {
            let gp = dir.join("ground_height.f64");
            let bytes: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&gp, bytes).map_err(|e| Error::file(&gp, e.to_string()))?;
        }
        grid.save_manifest(false)?;
        Ok(BlockWriter { grid: Mutex::new(grid.clone()) })
    }

    /// Write one finished block; returns false (and writes nothing) for a
    /// block no pulse reached.
    pub fn write_block(&self, w: &Window, part: &RayVoxels) -> Result<bool> {
        let empty = part.f.iter().all(|v| v.iter().all(|&x| x == 0.0))
            && part.i.iter().all(|v| v.iter().all(|&x| x == 0))
            && part.ppl_lambda.as_ref().is_none_or(|v| v.iter().all(|&x| x == -1.0))
            && part.subvoxel_counts.as_ref().is_none_or(|v| v.iter().all(|&x| x == 0));
        if empty {
            return Ok(false);
        }
        let (dir, block) = {
            let g = self.grid.lock().unwrap_or_else(|e| e.into_inner());
            (g.dir.clone(), g.block)
        };
        let b: [usize; 3] = std::array::from_fn(|k| w.lo[k] / block[k]);
        write_block_file(&dir.join(block_name(b)), part)?;
        self.grid.lock().unwrap_or_else(|e| e.into_inner()).present.insert(b);
        Ok(true)
    }

    pub fn finish(&self, has_leaf: bool, has_wood: bool) -> Result<()> {
        let mut g = self.grid.lock().unwrap_or_else(|e| e.into_inner());
        g.has_leaf = has_leaf;
        g.has_wood = has_wood;
        g.save_manifest(true)
    }
}

/// Columns of a block file: the switched-on single-precision sums, the
/// counts, `ppl_lambda` and `subvoxel_counts`.
fn schema(params: &VoxelParams, part_f_on: &[bool]) -> String {
    let mut s = String::from("message sylva_voxel_block {\n");
    for (fld, on) in F::ALL.iter().zip(part_f_on) {
        if *on {
            s += &format!("  required float {};\n", fld.name());
        }
    }
    for fld in I::ALL {
        s += &format!("  required int32 {};\n", fld.name());
    }
    if params.attenuation.contains(&Attenuation::Ppl) {
        s += "  required float ppl_lambda;\n";
    }
    let n_sub = params.subvoxel_split.pow(3);
    if n_sub > 0 {
        s += &format!("  required fixed_len_byte_array({n_sub}) subvoxel_counts;\n");
    }
    s + "}\n"
}

fn write_block_file(path: &Path, part: &RayVoxels) -> Result<()> {
    let on: Vec<bool> = part.f.iter().map(|v| !v.is_empty()).collect();
    let schema = Arc::new(parse_message_type(&schema(&part.params, &on))?);
    let mut props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(1)?))
        .set_key_value_metadata(Some(vec![KeyValue::new("sylva.voxel_block".into(), VERSION.to_string())]));
    for fld in F::ALL {
        props = props.set_column_encoding(fld.name().into(), Encoding::BYTE_STREAM_SPLIT).set_column_dictionary_enabled(fld.name().into(), false);
    }
    props = props.set_column_encoding("ppl_lambda".into(), Encoding::BYTE_STREAM_SPLIT).set_column_dictionary_enabled("ppl_lambda".into(), false);
    let tmp = path.with_extension("parquet.tmp");
    let file = File::create(&tmp).map_err(|e| Error::file(&tmp, e.to_string()))?;
    let mut writer = SerializedFileWriter::new(file, schema, Arc::new(props.build()))?;
    let mut rg = writer.next_row_group()?;
    macro_rules! next {
        () => {
            rg.next_column()?.ok_or_else(|| Error::invalid("parquet: schema has fewer columns than the data"))?
        };
    }
    for v in part.f.iter().filter(|v| !v.is_empty()) {
        let mut c = next!();
        c.typed::<FloatType>().write_batch(v, None, None)?;
        c.close()?;
    }
    for v in &part.i {
        let mut c = next!();
        c.typed::<Int32Type>().write_batch(v, None, None)?;
        c.close()?;
    }
    if let Some(v) = &part.ppl_lambda {
        let mut c = next!();
        c.typed::<FloatType>().write_batch(v, None, None)?;
        c.close()?;
    }
    if let Some(v) = &part.subvoxel_counts {
        let n_sub = part.params.subvoxel_split.pow(3);
        let rows: Vec<FixedLenByteArray> = v.chunks_exact(n_sub).map(|c| FixedLenByteArray::from(ByteArray::from(c.to_vec()))).collect();
        let mut c = next!();
        c.typed::<FixedLenByteArrayType>().write_batch(&rows, None, None)?;
        c.close()?;
    }
    rg.close()?;
    writer.close()?;
    std::fs::rename(&tmp, path).map_err(|e| Error::file(path, e.to_string()))
}

fn read_block(path: &Path, params: &VoxelParams, shape: [usize; 3]) -> Result<RayVoxels> {
    let file = File::open(path).map_err(|e| Error::file(path, e.to_string()))?;
    let reader = SerializedFileReader::new(file)?;
    let n = shape[0] * shape[1] * shape[2];
    let mut out = empty_grid(params, [0.0; 3], shape);
    let names: Vec<String> = reader.metadata().file_metadata().schema_descr().columns().iter().map(|c| c.name().to_string()).collect();
    let bad = |what: &str| Error::file(path, format!("voxel block: {what}"));
    let mut at = vec![0usize; names.len()];
    for g in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(g)?;
        let rows = rg.metadata().num_rows() as usize;
        for (c, name) in names.iter().enumerate() {
            let start = at[c];
            if start + rows > n {
                return Err(bad("more rows than the block has voxels"));
            }
            macro_rules! read {
                ($variant:ident, $ty:ty, $dst:expr) => {{
                    let ColumnReader::$variant(mut r) = rg.get_column_reader(c)? else { return Err(bad("unexpected column type")) };
                    let mut values: Vec<$ty> = Vec::with_capacity(rows);
                    let mut done = 0;
                    while done < rows {
                        let (records, _, _) = r.read_records(rows - done, None, None, &mut values)?;
                        if records == 0 {
                            break;
                        }
                        done += records;
                    }
                    if values.len() != rows {
                        return Err(bad("short column"));
                    }
                    values
                }};
            }
            if let Some(fld) = F::ALL.iter().find(|f| f.name() == name) {
                let v = read!(FloatColumnReader, f32, ());
                let dst = &mut out.f[*fld as usize];
                if dst.is_empty() {
                    return Err(bad("column the tracing options do not switch on"));
                }
                dst[start..start + rows].copy_from_slice(&v);
            } else if let Some(fld) = I::ALL.iter().find(|f| f.name() == name) {
                let v = read!(Int32ColumnReader, i32, ());
                out.i[*fld as usize][start..start + rows].copy_from_slice(&v);
            } else if name == "ppl_lambda" {
                let v = read!(FloatColumnReader, f32, ());
                out.ppl_lambda.as_mut().ok_or_else(|| bad("unexpected ppl_lambda"))?[start..start + rows].copy_from_slice(&v);
            } else if name == "subvoxel_counts" {
                let v = read!(FixedLenByteArrayColumnReader, FixedLenByteArray, ());
                let n_sub = params.subvoxel_split.pow(3);
                let dst = out.subvoxel_counts.as_mut().ok_or_else(|| bad("unexpected subvoxel_counts"))?;
                for (r, row) in v.iter().enumerate() {
                    dst[(start + r) * n_sub..(start + r + 1) * n_sub].copy_from_slice(row.data());
                }
            } else {
                return Err(bad(&format!("unknown column {name:?}")));
            }
            at[c] += rows;
        }
    }
    if at.iter().any(|&a| a != n) {
        return Err(bad("fewer rows than the block has voxels"));
    }
    Ok(out)
}
