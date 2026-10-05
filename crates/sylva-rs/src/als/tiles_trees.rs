// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Trees over tiles: segmentation whose labels are those of the whole plot,
//! stores of each tree's points, and per-tree values written back to the
//! tiles.
//!
//! [`segment_trees`] runs the segmentation of [`crate::trees`]
//! ([`trees::merge_branches`], [`trees::segment_trees`],
//! [`trees::tree_heights`] and [`crate::trees::prune::prune_trees`]) over a
//! catalogue of height-normalised tiles, holding a few tiles at a time:
//!
//! 1. The graph nodes of the whole plot (the points above
//!    `cut_above_ground`, thinned to the first point of each voxel of one
//!    global grid, in catalogue order) are found tile by tile and kept in
//!    scratch files. They are the nodes [`trees::segment_trees`] builds its
//!    graph on for the whole cloud when its grid has the same corner
//!    ([`SegmentParams::voxel_origin`]).
//! 2. Branches are merged ([`trees::branch_parents`]) on each tile's nodes
//!    with a buffer, each candidate traced by the tile whose core holds it,
//!    and the chains resolved over all candidates ([`trees::resolve_branches`]).
//! 3. Each tile's nodes and those of its buffer are segmented
//!    ([`trees::segment_nodes`]) with every stem that could reach them; the
//!    tile keeps the labels of the trees whose stem it holds. A tree whose
//!    nodes come within `edge_margin` of the buffer's edge may have been cut
//!    short or lost a contest it would win in the whole plot; the tile is run
//!    again with a buffer twice as wide, up to `max_buffer`, and trees still
//!    at the edge are reported.
//! 4. Every point takes the label of its nearest node ([`trees::label_points`]),
//!    tile by tile: the nearest node lies in the point's own voxel or closer,
//!    so a buffer of one voxel diagonal holds it.
//! 5. Heights and point counts per tree ([`trees::tree_heights`]), pruning
//!    over the tree list, and the tiles written with the final ids.
//!
//! [`split_trees`] then writes each tree's points (from every tile it
//! touches, in catalogue order: the order the whole cloud has them in) to a
//! store of per-tree files, so that one tree can be read without the plot;
//! [`write_tree_values`] keeps a value per point of a tree (leaf / wood, say)
//! beside it, and [`write_back`] writes such values into the tiles.

#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use crate::als::{self, merge_clouds, plan, run, Catalog, Layout, BYTES_PER_POINT};
use crate::error::{Error, Result};
use crate::io::las::read_las_where;
use crate::pointcloud::Attr;
use crate::util::spatial::voxel_key;
use crate::als::tiles::{self, files_in, margin, open_sides, owner, read_block, write_block, RunInfo, Scratch};
use crate::trees::prune::{prune_trees, PruneParams};
use crate::trees::{self, SegmentParams, Tree};
use crate::util::{limits, progress};
use crate::{Point, PointCloud};

/// Bytes held per graph node while a tile's nodes are segmented: the node,
/// its edges, the search tree and the path costs.
pub const BYTES_PER_NODE: u64 = 512;

// ------------------------------------------------------------------ helpers

#[derive(Debug, Default)]
struct Counts {
    chunks: AtomicUsize,
    max_points: AtomicUsize,
    points_read: AtomicU64,
    rereads: AtomicUsize,
    widened: AtomicU64,
}

impl Counts {
    fn read(&self, n: usize) {
        self.max_points.fetch_max(n, Ordering::Relaxed);
        self.points_read.fetch_add(n as u64, Ordering::Relaxed);
    }

    fn info(&self) -> RunInfo {
        RunInfo {
            chunks: self.chunks.load(Ordering::Relaxed),
            max_points: self.max_points.load(Ordering::Relaxed),
            points_read: self.points_read.load(Ordering::Relaxed),
            rereads: self.rereads.load(Ordering::Relaxed),
            widened_points: self.widened.load(Ordering::Relaxed),
        }
    }
}

fn grow(b: &[f64; 4], d: f64) -> [f64; 4] {
    [b[0] - d, b[1] - d, b[2] + d, b[3] + d]
}

#[inline]
fn in_box(b: &[f64; 4], p: &Point) -> bool {
    p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]
}

/// Run `f(i)` for `i` in `0..n` on `workers` threads, each taking the next
/// item when it is done with one; results in order. The first error stops
/// the workers taking new items and is returned.
fn run_each<T: Send>(n: usize, workers: usize, label: &str, f: impl Fn(usize) -> Result<T> + Sync) -> Result<Vec<T>> {
    let task = progress::start(label, n as u64);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let errors: Mutex<Vec<(usize, Error)>> = Mutex::new(Vec::new());
    let work = || loop {
        if failed.load(Ordering::Relaxed) {
            break;
        }
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        match f(i) {
            Ok(v) => results.lock().expect("results lock")[i] = Some(v),
            Err(e) => {
                failed.store(true, Ordering::Relaxed);
                errors.lock().expect("errors lock").push((i, e));
            }
        }
        task.inc(1);
    };
    let w = workers.max(1).min(n.max(1));
    if w == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..w {
                s.spawn(work);
            }
        });
    }
    let mut errors = errors.into_inner().expect("errors lock");
    errors.sort_by_key(|e| e.0);
    if let Some((_, e)) = errors.into_iter().next() {
        return Err(e);
    }
    Ok(results.into_inner().expect("results lock").into_iter().map(|r| r.expect("every item ran")).collect())
}

fn write_i64s(path: &Path, v: &[i64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn read_i64s(path: &Path) -> Result<Vec<i64>> {
    let bytes = fs::read(path)?;
    Ok(bytes.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect())
}

fn write_cloud(path: &Path, cloud: &PointCloud) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        write_block(&mut w, cloud)?;
        w.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn read_cloud(path: &Path) -> Result<PointCloud> {
    let mut r = BufReader::new(File::open(path)?);
    Ok(read_block(&mut r)?.unwrap_or_default())
}

/// Tile names in catalogue order, the key that ties scratch and store files
/// to tiles.
fn tile_name(cat: &Catalog, i: usize) -> String {
    cat.tiles[i].path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("tile_{i}"))
}

// ------------------------------------------------------------------ graph nodes

/// Nodes, their heights, and the tile and position each came from.
type NodeRead = (Vec<Point>, Vec<f64>, Vec<(u32, u32)>);

/// The graph nodes of a catalogue, one scratch file per tile.
struct Nodes {
    dir: PathBuf,
    count: Vec<usize>,
}

impl Nodes {
    fn path(&self, tile: usize) -> PathBuf {
        self.dir.join(format!("{tile:06}.pts"))
    }

    /// The nodes inside the closed box `outer`, in catalogue order, with
    /// their heights and the tile and position each came from.
    fn read(&self, cat: &Catalog, outer: &[f64; 4]) -> Result<NodeRead> {
        let (mut pts, mut h, mut src) = (Vec::new(), Vec::new(), Vec::new());
        for t in files_in(cat, outer) {
            if self.count[t] == 0 {
                continue;
            }
            let c = read_cloud(&self.path(t))?;
            let hh = c.attr("h").map(|a| a.to_f64()).unwrap_or_default();
            for (j, p) in c.xyz.iter().enumerate() {
                if in_box(outer, p) {
                    pts.push(*p);
                    h.push(hh[j]);
                    src.push((t as u32, j as u32));
                }
            }
        }
        Ok((pts, h, src))
    }

    /// Nodes in a box grown from the tiles' header extents.
    fn estimate(&self, cat: &Catalog, outer: &[f64; 4]) -> u64 {
        files_in(cat, outer).into_iter().map(|t| self.count[t] as u64).sum()
    }
}

/// The first point at or above `cut` of each voxel of the grid with a corner
/// at `origin`, in catalogue order (every such point if `voxel` is 0), with
/// its height, written per tile to `dir`.
fn build_nodes(cat: &Catalog, dir: &Path, height_attr: &str, voxel: f64, cut: f64, origin: Point, workers: usize, counts: &Counts) -> Result<Nodes> {
    fs::create_dir_all(dir)?;
    // Points of one voxel are less than a voxel apart; the margin covers rounding.
    let buffer = if voxel > 0.0 { voxel * (1.0 + 1e-6) + 1e-9 } else { 0.0 };
    let chunks = plan(cat, Layout::Tiles, buffer)?;
    let w = als::workers_for(&chunks, workers, BYTES_PER_POINT)?;
    let mut count = vec![0usize; cat.tiles.len()];
    let path = |t: usize| dir.join(format!("{t:06}.pts"));
    let made = run(cat, &chunks, w, "graph nodes", |chunk, data| {
        counts.read(data.cloud.len());
        let own = chunk.own.expect("one chunk per tile");
        let h = data.cloud.heights(height_attr);
        let mut seen = HashSet::new();
        let mut keep = Vec::new();
        for (i, (p, &hi)) in data.cloud.xyz.iter().zip(&h).enumerate() {
            // NaN heights are below every cut, as in trees::graph_nodes.
            if hi.is_nan() || hi < cut {
                continue;
            }
            if voxel > 0.0 && !seen.insert(voxel_key(p, &origin, voxel)) {
                continue;
            }
            if !data.buffer[i] {
                keep.push(i);
            }
        }
        let mut nodes = PointCloud::new(keep.iter().map(|&i| data.cloud.xyz[i]).collect());
        nodes.attrs.insert("h".into(), Attr::F64(keep.iter().map(|&i| h[i]).collect()));
        write_cloud(&path(own), &nodes)?;
        Ok((own, keep.len()))
    })?;
    for (own, n) in made.into_iter().flatten() {
        count[own] = n;
    }
    Ok(Nodes { dir: dir.to_path_buf(), count })
}

// ------------------------------------------------------------------ segmentation

/// The branch merging of [`trees::merge_branches`] with its own graph.
#[derive(Debug, Clone)]
pub struct MergeSettings {
    pub graph: SegmentParams,
    pub ground_height: f64,
    pub trunk_scale: f64,
    pub trunk_min: f64,
    pub search_radius: f64,
}

/// Settings of [`segment_trees`].
#[derive(Debug, Clone)]
pub struct TreeTiling {
    /// Attribute holding height above ground (z without it).
    pub height_attr: String,
    /// The segmentation. Its `voxel_origin` must be set: tiles share one grid.
    pub segment: SegmentParams,
    /// Merge branches into their stems first.
    pub merge: Option<MergeSettings>,
    /// Height percentile of [`trees::tree_heights`].
    pub percentile: f64,
    /// Prune the trees after segmentation.
    pub prune: Option<PruneParams>,
    /// Band read around each tile (m); wider than the largest crown.
    pub buffer: f64,
    /// Widest band a tile is read again with when a crown reaches the edge.
    pub max_buffer: f64,
    /// A tree whose nodes come this close (m) to the edge of the band, where
    /// the catalogue goes on, counts as reaching it.
    pub edge_margin: f64,
    /// Name of the label attribute written to the tiles.
    pub attribute: String,
}

impl Default for TreeTiling {
    fn default() -> Self {
        TreeTiling { height_attr: "height".into(), segment: SegmentParams { voxel_origin: Some([0.0; 3]), ..Default::default() }, merge: None, percentile: 100.0, prune: None, buffer: 20.0, max_buffer: 60.0, edge_margin: 2.0, attribute: "tree_id".into() }
    }
}

/// What [`segment_trees`] found and wrote.
#[derive(Debug, Clone, Default)]
pub struct Segmented {
    /// The final trees, each with the index of the input stem it came from.
    pub trees: Vec<(usize, Tree)>,
    /// Final ids of trees whose crown reached the edge of the widest band
    /// their tile was read with.
    pub at_edge: Vec<i64>,
    /// The tiles written.
    pub paths: Vec<PathBuf>,
    /// Graph nodes claimed by two tiles for different trees (0 when every
    /// tile's crowns lie within its buffer).
    pub conflicts: u64,
    pub info: RunInfo,
}

fn check_settings(s: &TreeTiling) -> Result<()> {
    for (name, v) in [("buffer", s.buffer), ("edge_margin", s.edge_margin)] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Error::invalid(format!("{name} must be a non-negative number of metres, got {v}")));
        }
    }
    if !(s.max_buffer.is_finite() && s.max_buffer >= s.buffer) {
        return Err(Error::invalid(format!("max_buffer must be at least the buffer ({}), got {}", s.buffer, s.max_buffer)));
    }
    if !(0.0..=100.0).contains(&s.percentile) {
        return Err(Error::invalid(format!("percentile must be within 0-100, got {}", s.percentile)));
    }
    let grids = std::iter::once(&s.segment).chain(s.merge.iter().map(|m| &m.graph));
    for g in grids {
        if g.voxel_origin.is_none_or(|o| o.iter().any(|v| !v.is_finite())) {
            return Err(Error::invalid("tiled segmentation needs a finite voxel_origin, shared by the tiles"));
        }
        if g.voxel_size.is_nan() || g.voxel_size < 0.0 || g.k == 0 {
            return Err(Error::invalid(format!("voxel_size must be non-negative and k positive, got {} and {}", g.voxel_size, g.k)));
        }
    }
    if s.attribute.is_empty() {
        return Err(Error::invalid("the label attribute needs a name"));
    }
    Ok(())
}

/// The graph settings for nodes already thinned: every node is used.
fn on_nodes(p: &SegmentParams) -> SegmentParams {
    SegmentParams { voxel_size: 0.0, ..p.clone() }
}

/// For every stem, the stem whose trunk its path to the ground runs through
/// (as [`trees::branch_parents`] on the whole plot), each traced by the tile
/// whose core holds it.
fn merge_parents(cat: &Catalog, nodes: &Nodes, stems: &[Tree], m: &MergeSettings, cores: &[[f64; 4]], buffer: f64, workers: usize, counts: &Counts) -> Result<Vec<Option<usize>>> {
    let g = on_nodes(&m.graph);
    let owners: Vec<usize> = stems.iter().map(|t| owner(cores, t.x, t.y)).collect();
    let est: Vec<u64> = cores.iter().map(|c| nodes.estimate(cat, &grow(c, buffer))).collect();
    let w = als::workers_for_estimates(&est, workers, BYTES_PER_NODE)?;
    let found = run_each(cores.len(), w, "merging branches", |ci| {
        if !owners.contains(&ci) {
            return Ok(Vec::new());
        }
        let outer = grow(&cores[ci], buffer);
        let (pts, h, _) = nodes.read(cat, &outer)?;
        counts.read(pts.len());
        let reach = grow(&cores[ci], m.search_radius + 1e-9);
        let local: Vec<usize> = (0..stems.len()).filter(|&i| in_box(&reach, &[stems[i].x, stems[i].y, 0.0])).collect();
        let trees_here: Vec<Tree> = local.iter().map(|&i| stems[i].clone()).collect();
        let only: Vec<bool> = local.iter().map(|&i| owners[i] == ci).collect();
        let parents = trees::branch_parents(&pts, &h, &trees_here, &g, m.ground_height, m.trunk_scale, m.trunk_min, m.search_radius, Some(&only));
        Ok(local.iter().zip(parents).filter(|(&i, _)| owners[i] == ci).map(|(&i, p)| (i, p.map(|j| local[j]))).collect::<Vec<_>>())
    })?;
    let mut parent = vec![None; stems.len()];
    for (i, p) in found.into_iter().flatten() {
        parent[i] = p;
    }
    Ok(parent)
}

/// How far from its stem a tree can have seeds, a height-prior node or an
/// understorey clearance, over all the trees.
fn stem_reach(stems: &[Tree], p: &SegmentParams) -> f64 {
    stems.iter().map(|t| {
        let d = if t.dbh.is_finite() { t.dbh } else { 0.0 };
        (0.5 * d + p.seed_radius).max(p.low_radius.max(1.5 * d)).max(p.height_prior_radius)
    }).fold(0.0, f64::max) + 1e-6
}

/// Segment tile `ci`'s nodes with its buffer, widening it while a tree of
/// the tile reaches the edge; write the labels of the tile's own trees as
/// claims on the nodes (`claims/<tile>_<chunk>.bin`). Returns the ids of the
/// tile's trees still at the edge.
fn segment_tile(cat: &Catalog, nodes: &Nodes, ci: usize, cores: &[[f64; 4]], stems: &[Tree], owners: &[usize], s: &TreeTiling, claims: &Path, counts: &Counts) -> Result<Vec<i64>> {
    let ext = cat.xy_bounds().ok_or_else(|| Error::invalid("the catalogue has no tiles"))?;
    let g = on_nodes(&s.segment);
    let reach = stem_reach(stems, &s.segment);
    let own: HashSet<i64> = (0..stems.len()).filter(|&i| owners[i] == ci).map(|i| stems[i].tree_id).collect();
    let mut b = s.buffer;
    loop {
        let outer = grow(&cores[ci], b);
        if b > s.buffer {
            let est = nodes.estimate(cat, &outer);
            limits::check(est.saturating_mul(BYTES_PER_NODE), &format!("tile {ci} read again with a {b} m buffer (about {est} graph nodes)"), "a smaller max_buffer")?;
        }
        let (pts, h, src) = nodes.read(cat, &outer)?;
        counts.read(pts.len());
        let near_box = grow(&outer, reach);
        let near: Vec<Tree> = stems.iter().filter(|t| in_box(&near_box, &[t.x, t.y, 0.0])).cloned().collect();
        let labels = trees::segment_nodes(&pts, &h, &near, &g);
        let open = open_sides(&ext, &outer);
        let mut edge: Vec<i64> = labels.iter().zip(&pts).filter(|(l, p)| own.contains(l) && margin(&outer, &open, p) < s.edge_margin).map(|(l, _)| *l).collect::<HashSet<_>>().into_iter().collect();
        edge.sort_unstable();
        if !edge.is_empty() && b < s.max_buffer {
            b = (b.max(s.edge_margin) * 2.0).min(s.max_buffer);
            counts.rereads.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if b > s.buffer {
            counts.widened.fetch_add(pts.len() as u64, Ordering::Relaxed);
        }
        let mut by_tile: BTreeMap<u32, Vec<(u32, i64)>> = BTreeMap::new();
        for (i, &l) in labels.iter().enumerate() {
            if own.contains(&l) {
                by_tile.entry(src[i].0).or_default().push((src[i].1, l));
            }
        }
        for (t, v) in by_tile {
            let mut buf = Vec::with_capacity(v.len() * 12);
            for (j, l) in v {
                buf.extend_from_slice(&j.to_le_bytes());
                buf.extend_from_slice(&l.to_le_bytes());
            }
            fs::write(claims.join(format!("{t:06}_{ci:06}.bin")), buf)?;
        }
        return Ok(edge);
    }
}

/// The label of every node of each tile: the claim of the tile whose core
/// holds the node's tree. A node claimed by two tiles (which happens only
/// where a crown reaches past a buffer) takes the claim of its own tile if
/// that tile made one, or else of the first. Returns the conflicts.
fn resolve_claims(nodes: &Nodes, claims: &Path, labels: &Path, chunk_of_tile: &[Option<usize>], workers: usize) -> Result<u64> {
    let mut files: BTreeMap<usize, Vec<(usize, PathBuf)>> = BTreeMap::new();
    for e in fs::read_dir(claims)? {
        let p = e?.path();
        let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if let Some((t, c)) = name.split_once('_') {
            if let (Ok(t), Ok(c)) = (t.parse::<usize>(), c.parse::<usize>()) {
                files.entry(t).or_default().push((c, p));
            }
        }
    }
    let tiles: Vec<usize> = (0..nodes.count.len()).filter(|&t| nodes.count[t] > 0).collect();
    let conflicts = AtomicU64::new(0);
    run_each(tiles.len(), workers, "resolving labels", |k| {
        let t = tiles[k];
        let mut lab = vec![-1i64; nodes.count[t]];
        let mut from = vec![usize::MAX; nodes.count[t]];
        if let Some(list) = files.get(&t) {
            let mut list = list.clone();
            list.sort();
            for (c, p) in list {
                let bytes = fs::read(&p)?;
                for rec in bytes.as_chunks::<12>().0 {
                    let j = u32::from_le_bytes(rec[..4].try_into().expect("4 bytes")) as usize;
                    let l = i64::from_le_bytes(rec[4..].try_into().expect("8 bytes"));
                    if j >= lab.len() {
                        return Err(Error::invalid("a scratch file of the segmentation is damaged"));
                    }
                    if lab[j] == -1 {
                        lab[j] = l;
                        from[j] = c;
                    } else if lab[j] != l {
                        conflicts.fetch_add(1, Ordering::Relaxed);
                        if Some(c) == chunk_of_tile[t] && from[j] != c {
                            lab[j] = l;
                            from[j] = c;
                        }
                    }
                }
            }
        }
        write_i64s(&labels.join(format!("{t:06}.i64")), &lab)
    })?;
    Ok(conflicts.into_inner())
}

/// Segment the trees of a catalogue of height-normalised tiles, as
/// [`trees::merge_branches`] (with `s.merge`), [`trees::segment_trees`],
/// [`trees::tree_heights`] and [`prune_trees`] (with `s.prune`) do on the
/// whole cloud, and write the tiles to `out_dir` with the tree id of every
/// point (-1 for none) in `s.attribute`.
///
/// Each tree is segmented by the tile whose core holds its stem ([`owner`]),
/// from that tile's graph nodes and those within `s.buffer`. When the
/// buffer is wider than the largest crown (so that every tree competing with
/// the tile's trees has its stem, seeds and crown in it), the labels are
/// those of the whole cloud, up to ties in nearest-neighbour searches. A
/// tile whose trees reach the edge of the buffer is read again with a buffer
/// twice as wide, up to `s.max_buffer`; trees still at the edge are returned
/// in [`Segmented::at_edge`].
pub fn segment_trees(cat: &Catalog, stems: &[Tree], out_dir: &Path, format: Option<&str>, s: &TreeTiling, workers: usize) -> Result<Segmented> {
    check_settings(s)?;
    let mut ids = HashSet::new();
    for t in stems {
        if t.tree_id < 0 || !ids.insert(t.tree_id) {
            return Err(Error::invalid(format!("stem ids must be unique and non-negative; {} is not", t.tree_id)));
        }
        if !(t.x.is_finite() && t.y.is_finite()) {
            return Err(Error::invalid(format!("stem {} has no position", t.tree_id)));
        }
    }
    let counts = Counts::default();
    let (chunks, _) = tiles::prepare(cat, Some(out_dir), format, 0.0, workers)?;
    let cores: Vec<[f64; 4]> = chunks.iter().map(|c| c.core).collect();
    let mut chunk_of_tile = vec![None; cat.tiles.len()];
    for c in &chunks {
        chunk_of_tile[c.own.expect("one chunk per tile")] = Some(c.index);
    }
    let scratch = Scratch::new(out_dir, ".sylva-segment")?;
    let seg_origin = s.segment.voxel_origin.expect("checked");
    let nodes = build_nodes(cat, &scratch.0.join("nodes"), &s.height_attr, s.segment.voxel_size.max(0.0), s.segment.cut_above_ground, seg_origin, workers, &counts)?;

    // Branches, on their own graph (its own voxel size and cut).
    let (survivors, input_index): (Vec<Tree>, Vec<usize>) = match &s.merge {
        Some(m) if !stems.is_empty() => {
            let same = m.graph.voxel_size == s.segment.voxel_size && m.graph.cut_above_ground == s.segment.cut_above_ground && m.graph.voxel_origin == s.segment.voxel_origin;
            let own_nodes;
            let mn = if same {
                &nodes
            } else {
                own_nodes = build_nodes(cat, &scratch.0.join("merge_nodes"), &s.height_attr, m.graph.voxel_size.max(0.0), m.graph.cut_above_ground, m.graph.voxel_origin.expect("checked"), workers, &counts)?;
                &own_nodes
            };
            let parent = merge_parents(cat, mn, stems, m, &cores, s.buffer, workers, &counts)?;
            let (kept, merged_into) = trees::resolve_branches(stems, &parent);
            let idx: Vec<usize> = (0..stems.len()).filter(|&i| merged_into[i] == stems[i].tree_id).collect();
            (kept, idx)
        }
        _ => (stems.to_vec(), (0..stems.len()).collect()),
    };

    // Graph labels, tile by tile, then resolved per tile of nodes.
    let owners: Vec<usize> = survivors.iter().map(|t| owner(&cores, t.x, t.y)).collect();
    let claims = scratch.0.join("claims");
    fs::create_dir_all(&claims)?;
    let est: Vec<u64> = cores.iter().map(|c| nodes.estimate(cat, &grow(c, s.buffer))).collect();
    let w = als::workers_for_estimates(&est, workers, BYTES_PER_NODE)?;
    let edges = run_each(cores.len(), w, "segmenting tiles", |ci| {
        if !owners.contains(&ci) {
            return Ok(Vec::new());
        }
        counts.chunks.fetch_add(1, Ordering::Relaxed);
        segment_tile(cat, &nodes, ci, &cores, &survivors, &owners, s, &claims, &counts)
    })?;
    let node_labels = scratch.0.join("labels");
    fs::create_dir_all(&node_labels)?;
    let conflicts = resolve_claims(&nodes, &claims, &node_labels, &chunk_of_tile, workers)?;
    let _ = fs::remove_dir_all(&claims);

    // Every point from its nearest node; heights gathered per tree.
    let voxel = s.segment.voxel_size.max(0.0);
    let reach = voxel * 3f64.sqrt() * (1.0 + 1e-9) + 1e-6;
    let raw = scratch.0.join("raw");
    let heights_dir = scratch.0.join("heights");
    fs::create_dir_all(&raw)?;
    fs::create_dir_all(&heights_dir)?;
    let (chunks0, w0) = tiles::prepare(cat, None, format, 0.0, workers)?;
    run(cat, &chunks0, w0, "labelling points", |chunk, data| {
        let own = chunk.own.expect("one chunk per tile");
        let core = data.cloud.take(&data.core_indices());
        drop(data);
        counts.read(core.len());
        let h = core.heights(&s.height_attr);
        let outer = grow(&chunk.core, reach);
        let (npts, _, src) = nodes.read(cat, &outer)?;
        let mut cache: HashMap<u32, Vec<i64>> = HashMap::new();
        let mut nl = Vec::with_capacity(src.len());
        for &(t, j) in &src {
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(t) {
                e.insert(read_i64s(&node_labels.join(format!("{t:06}.i64")))?);
            }
            nl.push(cache[&t][j as usize]);
        }
        let labels = trees::label_points(&core.xyz, &h, &npts, &nl, &survivors, &s.segment);
        write_i64s(&raw.join(format!("{own:06}.i64")), &labels)?;
        let mut by_tree: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for (&l, &hh) in labels.iter().zip(&h) {
            if l >= 0 {
                by_tree.entry(l).or_default().push(hh);
            }
        }
        for (l, v) in by_tree {
            let d = heights_dir.join(l.to_string());
            fs::create_dir_all(&d)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            fs::write(d.join(format!("{own:06}.f64")), bytes)?;
        }
        Ok(())
    })?;

    // Heights and point counts, as tree_heights gives them on the whole cloud.
    let mut trees_out: Vec<Tree> = survivors.clone();
    let filled: Vec<Result<(f64, usize)>> = trees_out
        .par_iter()
        .map(|t| {
            let d = heights_dir.join(t.tree_id.to_string());
            let mut hs = Vec::new();
            if d.exists() {
                for e in fs::read_dir(&d)? {
                    let bytes = fs::read(e?.path())?;
                    hs.extend(bytes.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)));
                }
            }
            let labels = vec![t.tree_id; hs.len()];
            let mut one = [t.clone()];
            trees::tree_heights(&hs, &labels, &mut one, s.percentile);
            Ok((one[0].height, one[0].n_points))
        })
        .collect();
    for (t, r) in trees_out.iter_mut().zip(filled) {
        let (h, n) = r?;
        t.height = h;
        t.n_points = n;
    }

    // Pruning over the tree list: the ids each old id becomes.
    let max_id = trees_out.iter().map(|t| t.tree_id).max().unwrap_or(0).max(0);
    let (final_trees, lut): (Vec<(usize, Tree)>, Vec<i64>) = match &s.prune {
        Some(p) if !trees_out.is_empty() => {
            let all: Vec<i64> = (0..=max_id).collect();
            let (kept, lut) = prune_trees(&trees_out, &all, p)?;
            let mut n_new: HashMap<i64, usize> = HashMap::new();
            for t in &trees_out {
                let new = lut[t.tree_id as usize];
                if new >= 0 {
                    *n_new.entry(new).or_default() += t.n_points;
                }
            }
            let kept = kept.into_iter().map(|(i, mut t)| {
                t.n_points = n_new.get(&t.tree_id).copied().unwrap_or(0);
                (input_index[i], t)
            }).collect();
            (kept, lut)
        }
        _ => {
            let mut lut = vec![-1i64; max_id as usize + 1];
            for t in &trees_out {
                lut[t.tree_id as usize] = t.tree_id;
            }
            (trees_out.iter().enumerate().map(|(i, t)| (input_index[i], t.clone())).collect(), lut)
        }
    };
    let map = |l: i64| if l >= 0 && (l as usize) < lut.len() { lut[l as usize] } else { -1 };

    // The tiles, with the final ids.
    let (chunks_w, ww) = tiles::prepare(cat, Some(out_dir), format, 0.0, workers)?;
    let written = run(cat, &chunks_w, ww, "writing tiles", |chunk, data| {
        let own = chunk.own.expect("one chunk per tile");
        let mut core = data.cloud.take(&data.core_indices());
        drop(data);
        let raw_labels = read_i64s(&raw.join(format!("{own:06}.i64")))?;
        if raw_labels.len() != core.len() {
            return Err(Error::invalid(format!("tile {} changed while it was being segmented", chunk.name)));
        }
        let ids: Vec<i32> = raw_labels.iter().map(|&l| map(l) as i32).collect();
        core.attrs.insert(s.attribute.clone(), Attr::I32(ids));
        tiles::write_tile(cat, out_dir, format, chunk, &core)
    })?;
    drop(scratch);
    let mut at_edge: Vec<i64> = edges.into_iter().flatten().map(map).filter(|&l| l >= 0).collect::<HashSet<_>>().into_iter().collect();
    at_edge.sort_unstable();
    Ok(Segmented { trees: final_trees, at_edge, paths: written.into_iter().flatten().flatten().collect(), conflicts, info: counts.info() })
}

// ------------------------------------------------------------------ tree stores

/// One tree of a store: its points per tile, in catalogue order.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeEntry {
    pub tree_id: i64,
    pub n_points: usize,
    /// `[xmin, ymin, zmin, xmax, ymax, zmax]` of its points.
    pub bounds: [f64; 6],
    /// `(tile index in the store's tile list, points in that tile)`.
    pub parts: Vec<(usize, usize)>,
}

const INDEX: &str = "index.tsv";
const TILES: &str = "tiles.txt";
const VALUES_MAGIC: [u8; 4] = *b"SYTV";

fn tree_dir(store: &Path, tree_id: i64) -> PathBuf {
    store.join(format!("tree_{tree_id}"))
}

/// Write every tree's points from the tiles to a store in `out_dir`: a
/// directory per tree (`tree_<id>`) holding its points of each tile it
/// touches, at full precision and with every attribute, plus the list of
/// tiles (`tiles.txt`) and an index of the trees (`index.tsv`). Points
/// labelled below 0 in `attribute` belong to no tree. One tile per worker is
/// in memory.
pub fn split_trees(cat: &Catalog, out_dir: &Path, attribute: &str, workers: usize) -> Result<(Vec<TreeEntry>, RunInfo)> {
    let counts = Counts::default();
    fs::create_dir_all(out_dir)?;
    for e in fs::read_dir(out_dir)? {
        let p = e?.path();
        let stale = p.file_name().map(|n| n.to_string_lossy().starts_with("tree_")).unwrap_or(false);
        if stale && p.is_dir() {
            fs::remove_dir_all(&p)?;
        }
    }
    let _ = fs::remove_file(out_dir.join(INDEX));
    let (chunks, w) = tiles::prepare(cat, None, None, 0.0, workers)?;
    let found = run(cat, &chunks, w, "splitting trees", |chunk, data| {
        counts.chunks.fetch_add(1, Ordering::Relaxed);
        let own = chunk.own.expect("one chunk per tile");
        let core = data.cloud.take(&data.core_indices());
        drop(data);
        counts.read(core.len());
        let labels = core.attr(attribute).ok_or_else(|| Error::invalid(format!("tile {} has no {attribute:?} attribute", chunk.name)))?.to_f64();
        let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &l) in labels.iter().enumerate() {
            if l >= 0.0 {
                groups.entry(l as i64).or_default().push(i);
            }
        }
        let mut out = Vec::with_capacity(groups.len());
        for (id, idx) in groups {
            let part = core.take(&idx);
            let d = tree_dir(out_dir, id);
            fs::create_dir_all(&d)?;
            write_cloud(&d.join(format!("{own:06}.pts")), &part)?;
            let (lo, hi) = part.bounds().expect("non-empty");
            out.push((id, own, idx.len(), [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]));
        }
        Ok(out)
    })?;
    let mut entries: BTreeMap<i64, TreeEntry> = BTreeMap::new();
    for (id, t, n, b) in found.into_iter().flatten().flatten() {
        let e = entries.entry(id).or_insert(TreeEntry { tree_id: id, n_points: 0, bounds: [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], parts: Vec::new() });
        e.n_points += n;
        for k in 0..3 {
            e.bounds[k] = e.bounds[k].min(b[k]);
            e.bounds[k + 3] = e.bounds[k + 3].max(b[k + 3]);
        }
        e.parts.push((t, n));
    }
    let mut entries: Vec<TreeEntry> = entries.into_values().collect();
    for e in &mut entries {
        e.parts.sort_unstable();
    }
    let names: Vec<String> = (0..cat.tiles.len()).map(|i| tile_name(cat, i)).collect();
    fs::write(out_dir.join(TILES), names.join("\n") + "\n")?;
    let mut text = String::from("tree_id\tn_points\txmin\tymin\tzmin\txmax\tymax\tzmax\tparts\n");
    for e in &entries {
        let parts: Vec<String> = e.parts.iter().map(|(t, n)| format!("{t}:{n}")).collect();
        let b: Vec<String> = e.bounds.iter().map(|v| format!("{v:?}")).collect();
        text.push_str(&format!("{}\t{}\t{}\t{}\n", e.tree_id, e.n_points, b.join("\t"), parts.join(",")));
    }
    let tmp = out_dir.join("index.tmp");
    fs::write(&tmp, text)?;
    fs::rename(&tmp, out_dir.join(INDEX))?;
    Ok((entries, counts.info()))
}

/// The tile names and the trees of a store written by [`split_trees`].
pub fn read_store(store: &Path) -> Result<(Vec<String>, Vec<TreeEntry>)> {
    let bad = || Error::invalid(format!("{} is not a tree store (see split_trees)", store.display()));
    let tiles_text = fs::read_to_string(store.join(TILES)).map_err(|_| bad())?;
    let index = fs::read_to_string(store.join(INDEX)).map_err(|_| bad())?;
    let names: Vec<String> = tiles_text.lines().map(|s| s.to_string()).collect();
    let mut entries = Vec::new();
    for line in index.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 9 {
            return Err(bad());
        }
        let num = |s: &str| s.parse::<f64>().map_err(|_| bad());
        let mut bounds = [0.0; 6];
        for k in 0..6 {
            bounds[k] = num(f[2 + k])?;
        }
        let mut parts = Vec::new();
        for p in f[8].split(',').filter(|p| !p.is_empty()) {
            let (t, n) = p.split_once(':').ok_or_else(bad)?;
            parts.push((t.parse().map_err(|_| bad())?, n.parse().map_err(|_| bad())?));
        }
        entries.push(TreeEntry { tree_id: f[0].parse().map_err(|_| bad())?, n_points: f[1].parse().map_err(|_| bad())?, bounds, parts });
    }
    Ok((names, entries))
}

/// One tree's points from a store, in catalogue order, with every attribute.
pub fn read_tree(store: &Path, tree_id: i64) -> Result<PointCloud> {
    let d = tree_dir(store, tree_id);
    if !d.is_dir() {
        return Err(Error::invalid(format!("the store {} has no tree {tree_id}", store.display())));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(&d)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "pts")).collect();
    files.sort();
    let parts = files.iter().map(|p| read_cloud(p)).collect::<Result<Vec<_>>>()?;
    Ok(merge_clouds(parts))
}

/// Keep a value per point of one tree (in the order [`read_tree`] gives the
/// points) under `name`, written atomically.
pub fn write_tree_values(store: &Path, tree_id: i64, name: &str, values: &Attr) -> Result<()> {
    if name.is_empty() || name.contains(['/', '\\', '.']) {
        return Err(Error::invalid(format!("a value name must be a plain word, got {name:?}")));
    }
    let d = tree_dir(store, tree_id);
    if !d.is_dir() {
        return Err(Error::invalid(format!("the store {} has no tree {tree_id}", store.display())));
    }
    let mut buf = Vec::with_capacity(16 + values.len() * 8);
    buf.extend_from_slice(&VALUES_MAGIC);
    buf.push(tiles::attr_code(values));
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    tiles::put_attr(&mut buf, values);
    let tmp = d.join(format!("{name}.tmp"));
    fs::write(&tmp, buf)?;
    fs::rename(&tmp, d.join(format!("{name}.val")))?;
    Ok(())
}

/// The values [`write_tree_values`] kept, None if there are none.
pub fn read_tree_values(store: &Path, tree_id: i64, name: &str) -> Result<Option<Attr>> {
    let p = tree_dir(store, tree_id).join(format!("{name}.val"));
    if !p.exists() {
        return Ok(None);
    }
    let mut r = BufReader::new(File::open(&p)?);
    let mut head = [0u8; 13];
    r.read_exact(&mut head)?;
    if head[..4] != VALUES_MAGIC {
        return Err(Error::invalid(format!("{} is damaged", p.display())));
    }
    let n = u64::from_le_bytes(head[5..13].try_into().expect("8 bytes")) as usize;
    Ok(Some(tiles::get_attr(head[4], n, &mut r)?))
}

/// An attribute of `code`'s type (that of [`tiles::attr_code`]) from values.
fn attr_from_f64(code: u8, v: Vec<f64>) -> Result<Attr> {
    Ok(match code {
        0 => Attr::F64(v),
        1 => Attr::F32(v.into_iter().map(|x| x as f32).collect()),
        2 => Attr::I64(v.into_iter().map(|x| x as i64).collect()),
        3 => Attr::I32(v.into_iter().map(|x| x as i32).collect()),
        4 => Attr::U32(v.into_iter().map(|x| x as u32).collect()),
        5 => Attr::U16(v.into_iter().map(|x| x as u16).collect()),
        6 => Attr::U8(v.into_iter().map(|x| x as u8).collect()),
        7 => Attr::I8(v.into_iter().map(|x| x as i8).collect()),
        8 => Attr::Bool(v.into_iter().map(|x| x != 0.0).collect()),
        _ => return Err(Error::invalid(format!("unknown attribute type {code}"))),
    })
}

/// Write the tiles of `cat` (those the store was split from) to `out_dir`
/// with an attribute `name` holding, for the points of each tree, the values
/// kept under `name` in the store ([`write_tree_values`]), and `default`
/// for every other point. `code` is the attribute's type ([`tiles::attr_code`]).
pub fn write_back(cat: &Catalog, store: &Path, out_dir: &Path, attribute: &str, name: &str, default: f64, code: u8, format: Option<&str>, workers: usize) -> Result<(Vec<PathBuf>, RunInfo)> {
    attr_from_f64(code, Vec::new())?;
    let (names, entries) = read_store(store)?;
    let tile_of: HashMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let by_id: HashMap<i64, &TreeEntry> = entries.iter().map(|e| (e.tree_id, e)).collect();
    let counts = Counts::default();
    let (chunks, w) = tiles::prepare(cat, Some(out_dir), format, 0.0, workers)?;
    let written = run(cat, &chunks, w, "writing values", |chunk, data| {
        counts.chunks.fetch_add(1, Ordering::Relaxed);
        let own = chunk.own.expect("one chunk per tile");
        let mut core = data.cloud.take(&data.core_indices());
        drop(data);
        counts.read(core.len());
        let t = *tile_of.get(tile_name(cat, own).as_str()).ok_or_else(|| Error::invalid(format!("tile {} is not in the store {}", chunk.name, store.display())))?;
        let labels = core.attr(attribute).ok_or_else(|| Error::invalid(format!("tile {} has no {attribute:?} attribute", chunk.name)))?.to_f64();
        let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &l) in labels.iter().enumerate() {
            if l >= 0.0 {
                groups.entry(l as i64).or_default().push(i);
            }
        }
        let mut out = vec![default; core.len()];
        for (id, idx) in groups {
            let e = by_id.get(&id).ok_or_else(|| Error::invalid(format!("tree {id} of tile {} is not in the store", chunk.name)))?;
            let Some(values) = read_tree_values(store, id, name)? else { continue };
            if values.len() != e.n_points {
                return Err(Error::invalid(format!("tree {id} has {} values of {name:?} for {} points", values.len(), e.n_points)));
            }
            let off: usize = e.parts.iter().filter(|(pt, _)| *pt < t).map(|(_, n)| n).sum();
            let n = e.parts.iter().find(|(pt, _)| *pt == t).map(|(_, n)| *n).unwrap_or(0);
            if n != idx.len() {
                return Err(Error::invalid(format!("tile {} holds {} points of tree {id}, the store {n}", chunk.name, idx.len())));
            }
            for (k, &i) in idx.iter().enumerate() {
                out[i] = values.get_f64(off + k);
            }
        }
        core.attrs.insert(name.to_string(), attr_from_f64(code, out)?);
        tiles::write_tile(cat, out_dir, format, chunk, &core)
    })?;
    Ok((written.into_iter().flatten().flatten().collect(), counts.info()))
}

/// Read the points of the catalogue inside `bounds` whose `attribute` is
/// `tree_id`, without a store: every tile meeting `bounds` is read.
pub fn read_tree_from_tiles(cat: &Catalog, tree_id: i64, attribute: &str, bounds: Option<[f64; 4]>) -> Result<PointCloud> {
    cat.check_usable()?;
    let b = bounds.unwrap_or([f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]);
    let mut parts = Vec::new();
    for t in files_in(cat, &b) {
        let c = read_las_where(&cat.tiles[t].path, |p| in_box(&b, p))?;
        let Some(l) = c.attr(attribute) else { continue };
        let idx: Vec<usize> = (0..c.len()).filter(|&i| l.get_f64(i) == tree_id as f64).collect();
        if !idx.is_empty() {
            parts.push(c.take(&idx));
        }
    }
    Ok(merge_clouds(parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::las::LasWriteOptions;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-tiletrees-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    /// Stems with round trunks, conical crowns and ground, over 30 x 30 m,
    /// with a `height` attribute and a `pid`.
    fn forest() -> (PointCloud, Vec<Tree>) {
        let mut rng = crate::util::nprandom::Generator::new(7);
        let pos = [(5.0, 6.0), (14.0, 5.0), (24.0, 7.0), (6.0, 15.0), (16.0, 16.0), (25.0, 17.0), (7.0, 25.0), (15.0, 24.5), (24.0, 26.0), (19.5, 10.5)];
        let mut xyz = Vec::new();
        let mut stems = Vec::new();
        for (k, &(x, y)) in pos.iter().enumerate() {
            let r = 0.12 + 0.02 * k as f64;
            let top = 10.0 + k as f64;
            let u = rng.uniform_n(0.0, 1.0, 12_000);
            for i in 0..3000 {
                let a = std::f64::consts::TAU * u[i];
                let z = top * 0.6 * u[3000 + i];
                xyz.push([x + r * a.cos(), y + r * a.sin(), z]);
            }
            for i in 0..3000 {
                let z = top * (0.4 + 0.6 * u[6000 + i]);
                let rad = 3.0 * (top - z) / (0.6 * top) * u[9000 + i].sqrt();
                let a = std::f64::consts::TAU * u[3000 + i] * 7.0;
                xyz.push([x + rad * a.cos(), y + rad * a.sin(), z]);
            }
            stems.push(Tree { tree_id: k as i64 + 1, x, y, dbh: 2.0 * r, height: f64::NAN, n_points: 0, inlier_fraction: 1.0, n_slices: 5, rmse: 0.0, lean_deg: 0.0, quality: 1.0 - 0.01 * k as f64 });
        }
        let g = rng.uniform_n(0.0, 30.0, 40_000);
        for i in 0..20_000 {
            xyz.push([g[2 * i], g[2 * i + 1], 0.01 * ((i % 7) as f64)]);
        }
        let n = xyz.len();
        let mut c = PointCloud::new(xyz);
        let h: Vec<f64> = c.xyz.iter().map(|p| p[2]).collect();
        c.attrs.insert("height".into(), Attr::F64(h));
        c.attrs.insert("pid".into(), Attr::I64((0..n as i64).collect()));
        (c, stems)
    }

    fn tiled(cloud: &PointCloud, dir: &Path, size: f64) -> (Catalog, PointCloud) {
        let opts = LasWriteOptions { point_format: 6, scale: 0.0001, ..Default::default() };
        let written = crate::als::ops::write_tiles(cloud, dir, size, Some((0.0, 0.0)), "laz", &opts, None).unwrap();
        let cat = Catalog::open(&written.iter().map(|w| w.0.clone()).collect::<Vec<_>>());
        let whole = als::read_region(&cat, [-1e9, -1e9, 1e9, 1e9]).unwrap();
        (cat, whole)
    }

    fn by_pid(c: &PointCloud, attr: &str) -> BTreeMap<i64, f64> {
        let p = c.attr("pid").unwrap().to_f64();
        let v = c.attr(attr).unwrap().to_f64();
        p.iter().zip(v).map(|(a, b)| (*a as i64, b)).collect()
    }

    #[test]
    fn tiled_segmentation_matches_the_whole_cloud() {
        let d = tmp("seg");
        let (cloud, stems) = forest();
        let (cat, whole) = tiled(&cloud, &d.join("in"), 10.0);
        let h = whole.heights("height");
        let seg = SegmentParams { voxel_size: 0.1, voxel_origin: Some([0.0; 3]), understorey_height: 0.0, ..Default::default() };
        let merge = MergeSettings { graph: SegmentParams { k: 10, voxel_size: 0.1, seed_radius: 0.5, power: 3.0, height_prior: false, voxel_origin: Some([0.0; 3]), ..Default::default() }, ground_height: 0.5, trunk_scale: 1.5, trunk_min: 0.15, search_radius: 6.0 };
        let prune = PruneParams { min_height: 2.0, ..Default::default() };
        // The whole cloud.
        let (kept, _) = trees::merge_branches(&whole.xyz, &h, &stems, &merge.graph, 0.5, 1.5, 0.15, 6.0);
        let labels = trees::segment_trees(&whole.xyz, &h, &kept, &seg);
        let mut kept = kept;
        trees::tree_heights(&h, &labels, &mut kept, 99.0);
        let (expect, expect_labels) = prune_trees(&kept, &labels, &prune).unwrap();
        assert!(expect.len() >= 8, "{}", expect.len());
        for (workers, buffer) in [(1, 20.0), (3, 12.0)] {
            let s = TreeTiling { segment: seg.clone(), merge: Some(merge.clone()), percentile: 99.0, prune: Some(prune.clone()), buffer, max_buffer: 40.0, ..Default::default() };
            let out = d.join(format!("out{workers}"));
            let got = segment_trees(&cat, &stems, &out, None, &s, workers).unwrap();
            assert_eq!(got.conflicts, 0);
            assert!(got.at_edge.is_empty(), "{:?}", got.at_edge);
            let gt: Vec<Tree> = got.trees.iter().map(|(_, t)| t.clone()).collect();
            let et: Vec<Tree> = expect.iter().map(|(_, t)| t.clone()).collect();
            assert_eq!(gt, et);
            let back = merge_clouds(got.paths.iter().map(|p| crate::io::read(p).unwrap()).collect());
            let got_l = by_pid(&back, "tree_id");
            let mut want = whole.clone();
            want.attrs.insert("tree_id".into(), Attr::I64(expect_labels.clone()));
            assert_eq!(got_l, by_pid(&want, "tree_id"));
            assert!(!out.join(".sylva-segment").exists());
            // A store of the trees reads each back as the whole cloud has it.
            let store = d.join(format!("store{workers}"));
            let (entries, _) = split_trees(&Catalog::open(&got.paths), &store, "tree_id", workers).unwrap();
            assert_eq!(entries.len(), expect.len());
            for e in &entries {
                let t = read_tree(&store, e.tree_id).unwrap();
                let idx: Vec<usize> = (0..whole.len()).filter(|&i| expect_labels[i] == e.tree_id).collect();
                assert_eq!(t.xyz, whole.take(&idx).xyz);
                assert_eq!(t.len(), e.n_points);
            }
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn values_go_back_to_the_tiles() {
        let d = tmp("values");
        let (mut cloud, _) = forest();
        let labels: Vec<i32> = cloud.xyz.iter().map(|p| if p[2] > 0.5 { (p[0] / 10.0) as i32 + 3 * (p[1] / 10.0) as i32 } else { -1 }).collect();
        cloud.attrs.insert("tree_id".into(), Attr::I32(labels));
        let (cat, whole) = tiled(&cloud, &d.join("in"), 7.0);
        let (entries, _) = split_trees(&cat, &d.join("store"), "tree_id", 2).unwrap();
        for e in &entries {
            let t = read_tree(&d.join("store"), e.tree_id).unwrap();
            let v: Vec<i8> = t.xyz.iter().map(|p| (p[2] > 5.0) as i8).collect();
            write_tree_values(&d.join("store"), e.tree_id, "wood", &Attr::I8(v)).unwrap();
        }
        let (paths, _) = write_back(&cat, &d.join("store"), &d.join("out"), "tree_id", "wood", -1.0, 7, None, 3).unwrap();
        let back = merge_clouds(paths.iter().map(|p| crate::io::read(p).unwrap()).collect());
        let got = by_pid(&back, "wood");
        let l = whole.attr("tree_id").unwrap().to_f64();
        let p = whole.attr("pid").unwrap().to_f64();
        for i in 0..whole.len() {
            let want = if l[i] < 0.0 { -1.0 } else { (whole.xyz[i][2] > 5.0) as i8 as f64 };
            assert_eq!(got[&(p[i] as i64)], want);
        }
        assert!(write_tree_values(&d.join("store"), 999, "wood", &Attr::I8(vec![])).is_err());
        assert!(read_tree(&d.join("store"), 999).is_err());
        let one = read_tree_from_tiles(&cat, entries[0].tree_id, "tree_id", None).unwrap();
        assert_eq!(one.xyz, read_tree(&d.join("store"), entries[0].tree_id).unwrap().xyz);
        let _ = fs::remove_dir_all(&d);
    }
}
