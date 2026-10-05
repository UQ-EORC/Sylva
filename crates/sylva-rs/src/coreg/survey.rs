// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The coregistration pipeline: the whole survey.
//!
//! [`coregister_prepared`] screens candidate pairs by stem matching,
//! registers the survivors ([`crate::coreg::pipeline::register_pair`]) on a
//! given number of threads, refuses pairs that contradict approximate poses,
//! and makes the accepted pairs the edges of a pose graph, each weighted by
//! the directions its surfaces constrain. The graph is solved with outlier
//! rejection; scans left over are retried against the combined registered
//! survey and tied in by pairwise measurements, or placed from their priors
//! ([`place_from_prior`]); optionally every pose is refined jointly.
//! [`coregister`] prepares the scans first, [`merge_clouds`] applies the
//! result, and the [`SurveyResult`] functions give its report, its stem
//! consistency and its `transforms.json`.
//!
//! Results never depend on the number of threads: work is handed out in
//! order and collected in order ([`crate::util::relay::ordered_map`]), and every
//! message is logged in the order the Python package logged it.

use std::path::Path;
use std::time::Instant;

use nalgebra::{Matrix4, Matrix6};

use crate::coreg::matching::{match_stem_maps, StemMatch};
use crate::coreg::icp::{icp, IcpResult};
use crate::coreg::pipeline::{
    above_ground_fitness, edge_information, edge_quality, ground_disagrees, height_offset, matmul, on_ground, prepare_scan, prior_ok, py_head, read_scan, register_pair, stem_map, transform_stems, unusable_scan, CoregConfig, PairResult, PrepareError, ScanFeatures, ScanInput, TargetCache,
};
use crate::coreg::posegraph as pg;
use crate::coreg::reflectors::Reflector;
use crate::coreg::stemmap::StemRecord;
use crate::coreg::transforms::{invert, transform_points, Mat4};
use crate::error::{Error, Result};
use crate::util::json::{self, Json};
use crate::util::numeric::{median, pairwise_sum};
use crate::pointcloud::{Attr, PointCloud};
use crate::util::pyformat::{boolean, fixed, fixed_width, general, left, signed, thousands};
use crate::util::relay::{ordered_map, Log};
use crate::Point;

/// Outcome of coregistering a survey (`SurveyResult`, without its scans).
#[derive(Debug, Clone)]
pub struct SurveyResult {
    pub pairs: Vec<PairResult>,
    /// `world_from_levelled_scan` per scan.
    pub poses: Vec<Mat4>,
    pub reference: usize,
    pub optimisation: Option<pg::Optimisation>,
    /// Per scan, connected to an anchor by accepted pairs.
    pub registered: Vec<bool>,
    pub seconds: f64,
    /// Index into `pairs` of each pose-graph edge.
    pub edge_to_pair: Vec<usize>,
}

/// What the survey-level functions ask of the scans beyond their poses.
#[derive(Debug, Clone, Default)]
pub struct SurveyOptions {
    /// Pairs to try; every usable pair within `max_pair_distance` if `None`.
    pub pairs: Option<Vec<(usize, usize)>>,
    /// Rough scanner positions, one row per scan (2 or more columns; NaN
    /// unknown); taken from `priors` if `None`.
    pub approximate_positions: Option<Vec<Vec<f64>>>,
    /// `world_from_levelled_scan` of scans held fixed, in the order given.
    pub fixed: Vec<(usize, Mat4)>,
    /// Per scan, an approximate `world_from_levelled_scan`.
    pub priors: Option<Vec<Option<Mat4>>>,
}

// ------------------------------------------------------------------ workers

/// Available memory (GB) from `/proc/meminfo`.
fn available_gb() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable"))?;
    let kb: i64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb as f64 / (1u64 << 20) as f64)
}

/// Workers to use (`_resolve_workers`): `requested` if positive, else the
/// cores, fewer if free memory allows fewer than one per
/// `memory_per_worker_gb` (0 leaves memory out); never more than the tasks.
pub fn resolve_workers(requested: i64, memory_per_worker_gb: f64, n_tasks: usize) -> usize {
    if n_tasks <= 1 {
        return 1;
    }
    if requested > 0 {
        return (requested as usize).min(n_tasks);
    }
    let mut workers = std::thread::available_parallelism().map_or(1, |n| n.get()) as i64;
    if memory_per_worker_gb != 0.0 {
        if let Some(available) = available_gb() {
            let per = (available / memory_per_worker_gb).floor();
            if per.is_finite() {
                workers = workers.min((per as i64).max(1));
            }
        }
    }
    workers.min(n_tasks as i64).max(1) as usize
}

// -------------------------------------------------------------------- graph

/// The survey's pose graph (`PoseGraph`): poses, edges and anchors.
struct Graph {
    n: usize,
    reference: usize,
    fixed: Vec<(usize, Mat4)>,
    poses: Vec<Mat4>,
    edges: Vec<pg::Edge>,
}

impl Graph {
    fn new(n: usize, reference: usize, fixed: &[(usize, Mat4)]) -> Result<Self> {
        if n < 1 {
            return Err(Error::invalid("a pose graph needs at least one node"));
        }
        if reference >= n {
            return Err(Error::invalid("reference node index out of range"));
        }
        let mut poses = vec![Matrix4::identity(); n];
        for (k, p) in fixed {
            if *k >= n {
                return Err(Error::invalid("list assignment index out of range"));
            }
            poses[*k] = *p;
        }
        Ok(Graph { n, reference, fixed: fixed.to_vec(), poses, edges: Vec::new() })
    }

    fn is_fixed(&self, k: usize) -> bool {
        self.fixed.iter().any(|(m, _)| *m == k)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_edge(&mut self, i: i64, j: i64, transform: &Mat4, information: Option<Matrix6<f64>>, fitness: f64, rmse: f64, n_correspondences: i64) -> Result<()> {
        let n = self.n as i64;
        if !(0 <= i && i < n && 0 <= j && j < n) {
            return Err(Error::invalid("edge endpoints out of range"));
        }
        if i == j {
            return Err(Error::invalid("self-edges are not allowed"));
        }
        let information = information.unwrap_or_else(|| pg::default_information(rmse, fitness, n_correspondences, 15.0));
        self.edges.push(pg::Edge { i: i as usize, j: j as usize, transform: *transform, information, weight: fitness * n_correspondences.max(1) as f64 });
        Ok(())
    }

    fn initialise(&mut self) {
        self.poses = pg::initialise(self.n, &self.edges, self.reference, &self.fixed);
    }

    fn optimise(&mut self, reject_outliers: bool) -> pg::Optimisation {
        if self.edges.is_empty() {
            return pg::Optimisation { poses: self.poses.clone(), iterations: 0, converged: true, initial_error: 0.0, final_error: 0.0, rejected_edges: Vec::new(), edge_errors: Vec::new() };
        }
        let params = pg::OptimiseParams { reject_outliers, ..pg::OptimiseParams::default() };
        let r = pg::optimise(self.n, &self.edges, self.reference, &self.fixed, self.poses.clone(), &params);
        self.poses = r.poses.clone();
        r
    }

    /// Scans connected to an anchor by accepted edges, rejected ones
    /// included (`_registered_mask`).
    fn registered(&self) -> Vec<bool> {
        let mut adjacency = vec![Vec::new(); self.n];
        for e in &self.edges {
            adjacency[e.i].push(e.j);
            adjacency[e.j].push(e.i);
        }
        let mut seen = vec![false; self.n];
        let mut stack = vec![self.reference];
        seen[self.reference] = true;
        for (k, _) in &self.fixed {
            if !seen[*k] {
                seen[*k] = true;
                stack.push(*k);
            }
        }
        while let Some(node) = stack.pop() {
            for &nb in &adjacency[node] {
                if !seen[nb] {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        seen
    }

    /// Keep `reference` unless it is isolated, then move it into the largest
    /// block (`_reference_in_largest_component`).
    fn reference_in_largest_component(&self, reference: usize) -> usize {
        let mut components = pg::components(self.n, &self.edges);
        components.sort_by_key(|c| (std::cmp::Reverse(c.len()), c[0]));
        if components.is_empty() || components[0].len() < 2 {
            return reference;
        }
        let mine = components.iter().find(|c| c.contains(&reference)).expect("every node has a component");
        if mine.len() > 1 {
            reference
        } else {
            components[0][0]
        }
    }
}

/// `OptimisationResult.__repr__`.
pub fn optimisation_repr(o: &pg::Optimisation) -> String {
    format!("OptimisationResult(iterations={}, converged={}, error {} -> {}, rejected={})", o.iterations, boolean(o.converged), general(o.initial_error, 4), general(o.final_error, 4), o.rejected_edges.len())
}

// ------------------------------------------------------------------ screening

/// Pairs within reach of each other's approximate positions (`_within_reach`);
/// an unknown position keeps the pair.
pub fn within_reach(candidates: &[(usize, usize)], positions: Option<&[Vec<f64>]>, limit: f64, log: Log) -> Result<Vec<(usize, usize)>> {
    let Some(positions) = positions.filter(|p| !p.is_empty()) else { return Ok(candidates.to_vec()) };
    if !limit.is_finite() {
        return Ok(candidates.to_vec());
    }
    let row = |k: usize| -> Result<[f64; 2]> {
        let r = positions.get(k).ok_or_else(|| Error::invalid(format!("index {k} is out of bounds for axis 0 with size {}", positions.len())))?;
        if r.len() < 2 {
            return Err(Error::invalid("approximate positions need at least two columns"));
        }
        Ok([r[0], r[1]])
    };
    let mut kept = Vec::new();
    for &(i, j) in candidates {
        let (a, b) = (row(i)?, row(j)?);
        if !(a.iter().chain(&b).all(|v| v.is_finite())) || (a[0] - b[0]).hypot(a[1] - b[1]) <= limit {
            kept.push((i, j));
        }
    }
    if kept.len() < candidates.len() {
        log(&format!("  {} of {} pairs skipped: more than {} m apart", thousands((candidates.len() - kept.len()) as i64), thousands(candidates.len() as i64), fixed(limit, 0)));
    }
    Ok(kept)
}

/// Each scan's strongest matches (`_limit_per_scan`): pairs `(i, j,
/// inliers)` taken strongest first, each kept if either end still has room;
/// the indices kept, in their original order.
pub fn limit_per_scan(pairs: &[(usize, usize, usize)], limit: i64, n_scans: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(pairs[k].2));
    let mut budget = vec![limit; n_scans];
    let mut chosen = Vec::new();
    for k in order {
        let (i, j, _) = pairs[k];
        if budget[i] > 0 || budget[j] > 0 {
            chosen.push(k);
            budget[i] -= 1;
            budget[j] -= 1;
        }
    }
    chosen.sort_unstable();
    chosen
}

type Screened = (Vec<((usize, usize), StemMatch)>, Vec<PairResult>);

/// Coarse-match every candidate pair; keep the matches worth ICP and the
/// results of those that are not (`_screen`).
fn screen(candidates: &[(usize, usize)], scans: &[ScanFeatures], cfg: &CoregConfig, workers: usize) -> Screened {
    let matched = ordered_map(candidates.len(), workers, |k| {
        let (i, j) = candidates[k];
        match_stem_maps(&scans[i].stem_map(), &scans[j].stem_map(), &cfg.matching)
    }, |_, _| {});
    let (mut kept, mut rejected) = (Vec::new(), Vec::new());
    for (&(i, j), m) in candidates.iter().zip(matched) {
        let mut probe = PairResult::new(i as i64, j as i64, &scans[i].name, &scans[j].name);
        probe.stem_match = Some(m.clone());
        let acceptable = crate::coreg::pipeline::screen_probe(&mut probe, &scans[i], &scans[j], &m, cfg);
        // Stems screen out a pair, but not its targets: those are tried in ICP.
        let targets = cfg.use_reflectors && (scans[i].reflectors.len().min(scans[j].reflectors.len()) as i64) >= cfg.min_reflector_matches.max(3);
        if acceptable || targets {
            kept.push(((i, j), m));
        } else {
            rejected.push(probe);
        }
    }
    if cfg.max_pairs_per_scan != 0 {
        let strength: Vec<(usize, usize, usize)> = kept.iter().map(|((i, j), m)| (*i, *j, m.n_inliers)).collect();
        let chosen = limit_per_scan(&strength, cfg.max_pairs_per_scan, scans.len());
        kept = chosen.into_iter().map(|k| kept[k].clone()).collect();
    }
    (kept, rejected)
}

// ----------------------------------------------------------------- recovery

/// Every registered scan's stems in the world frame, one per tree, best
/// first (`_combined_stem_map`): duplicates would wreck the one-to-one matcher.
pub fn combined_stem_map(scans: &[ScanFeatures], registered: &[bool], poses: &[Mat4]) -> Vec<StemRecord> {
    let mut stems = Vec::new();
    for (k, &ok) in registered.iter().enumerate() {
        if ok && !scans[k].stems.is_empty() {
            stems.extend(transform_stems(&scans[k].stems, &poses[k]));
        }
    }
    let q: Vec<f64> = stems.iter().map(|s| -s.quality()).collect();
    let mut order: Vec<usize> = (0..stems.len()).collect();
    order.sort_by(|&a, &b| q[a].partial_cmp(&q[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<StemRecord> = Vec::new();
    for k in order {
        let s = &stems[k];
        let near = keep.iter().map(|t| {
            let (dx, dy) = (t.x - s.x, t.y - s.y);
            (dx * dx + dy * dy).sqrt()
        });
        if near.fold(f64::INFINITY, f64::min) < 0.3 {
            continue;
        }
        keep.push(s.clone());
    }
    keep
}

fn distance3(a: &[f64], b: &[f64]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn sort_by_distance(items: &mut [usize], key: impl Fn(usize) -> f64) {
    items.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
}

fn merged_target(scans: &[&ScanFeatures], poses: &[Mat4]) -> Vec<Point> {
    scans.iter().zip(poses).flat_map(|(s, p)| transform_points(p, &s.icp_points)).collect()
}

type Placement = (Mat4, Vec<usize>, IcpResult);

/// A world pose for one scan from the combined survey, or `None`
/// (`_place_against_survey`).
fn place_against_survey(scans: &[ScanFeatures], k: usize, combined: &[StemRecord], registered: &[bool], poses: &[Mat4], cfg: &CoregConfig) -> Result<Option<Placement>> {
    let scan = &scans[k];
    if scan.stems.len() < 3 || (combined.len() as i64) < cfg.min_match_inliers {
        return Ok(None);
    }
    let m = match_stem_maps(&scan.stem_map(), &stem_map(combined), &cfg.matching);
    if !m.success || (m.n_inliers as i64) < cfg.min_match_inliers || m.ambiguity > cfg.max_match_ambiguity {
        return Ok(None); // an ambiguous match is a lattice: nothing to gain by guessing
    }
    let here = [m.transform.0[(0, 3)], m.transform.0[(1, 3)], m.transform.0[(2, 3)]];
    let mut others: Vec<usize> = (0..registered.len()).filter(|&o| registered[o] && !scans[o].icp_points.is_empty()).collect();
    if others.is_empty() {
        return Ok(None);
    }
    sort_by_distance(&mut others, |o| distance3(&[poses[o][(0, 3)], poses[o][(1, 3)], poses[o][(2, 3)]], &here));
    let neighbours = py_head(&others, cfg.recovery_neighbours.max(1));
    let around: Vec<(&ScanFeatures, Mat4)> = neighbours.iter().map(|&o| (&scans[o], poses[o])).collect();
    let target = merged_target(&around.iter().map(|a| a.0).collect::<Vec<_>>(), &around.iter().map(|a| a.1).collect::<Vec<_>>());
    let start = on_ground(&m.transform.0, scan, &around, cfg);
    let refined = icp(&scan.icp_points, &target, Some(start), &cfg.icp)?;
    if refined.fitness < cfg.min_icp_fitness || refined.inlier_rmse > cfg.max_icp_rmse {
        return Ok(None);
    }
    if above_ground_fitness(&scan.icp_points, &scan.icp_heights, &target, &refined.transform, cfg) < cfg.min_icp_fitness_above_ground {
        return Ok(None);
    }
    if ground_disagrees(height_offset(scan, &refined.transform, &around, cfg), cfg) {
        return Ok(None);
    }
    Ok(Some((refined.transform, neighbours, refined)))
}

/// Place one scan from an approximate pose instead of from stems
/// (`place_from_prior`).
///
/// The prior is first corrected in height by the median offset between the
/// scan's ground and that of the `neighbours` registered scans nearest it
/// (`recovery_neighbours` if `None` or 0); ICP then starts with a 1.5 m
/// correspondence distance. Accepted on ICP fitness, above-ground fitness,
/// RMSE and terrain, and only within `max_prior_shift` of the prior. The
/// result's `transform` is `world_from_scan`; the indices are into `survey`.
pub fn place_from_prior(scan: &ScanFeatures, survey: &[&ScanFeatures], poses: &[Mat4], prior: &Mat4, cfg: &CoregConfig, neighbours: Option<i64>) -> Result<(PairResult, Vec<usize>)> {
    let start = Instant::now();
    let mut result = PairResult::new(-1, -1, &scan.name, "survey");
    let here = scan.location(prior);
    let mut order: Vec<usize> = (0..survey.len()).collect();
    let located: Vec<Point> = (0..survey.len()).map(|m| survey[m].location(&poses[m])).collect();
    sort_by_distance(&mut order, |m| distance3(&located[m], &here));
    let with_points: Vec<usize> = order.into_iter().filter(|&m| !survey[m].icp_points.is_empty()).collect();
    let used = py_head(&with_points, neighbours.filter(|&n| n != 0).unwrap_or(cfg.recovery_neighbours));
    if used.is_empty() {
        result.reason = "no registered scans to place against".into();
        return Ok((result, Vec::new()));
    }
    let around: Vec<(&ScanFeatures, Mat4)> = used.iter().map(|&m| (survey[m], poses[m])).collect();
    let target = merged_target(&around.iter().map(|a| a.0).collect::<Vec<_>>(), &around.iter().map(|a| a.1).collect::<Vec<_>>());
    let mut coarse = *prior;
    let dz = height_offset(scan, &coarse, &around, cfg);
    if !dz.is_finite() {
        result.reason = "no ground shared with the registered scans".into();
        return Ok((result, Vec::new()));
    }
    coarse[(2, 3)] += dz;
    let mut wide = cfg.icp.clone();
    wide.voxel_sizes = vec![0.30, 0.30, 0.15, 0.07, 0.05];
    wide.max_distances = Some(vec![1.50, 0.80, 0.40, 0.20, 0.12]);
    let refined = icp(&scan.icp_points, &target, Some(coarse), &wide)?;
    result.transform = refined.transform;
    result.coarse_transform = coarse;
    result.fitness_above = above_ground_fitness(&scan.icp_points, &scan.icp_heights, &target, &refined.transform, cfg);
    let (good, why) = prior_ok(&refined.transform, prior, scan.origin, cfg);
    if refined.fitness < cfg.min_icp_fitness {
        result.reason = format!("low ICP fitness ({} < {})", fixed(refined.fitness, 3), crate::coreg::pipeline::py_float(cfg.min_icp_fitness));
    } else if result.fitness_above < cfg.min_icp_fitness_above_ground {
        result.reason = format!("low above-ground fitness ({}); ground alone matched", fixed(result.fitness_above, 3));
    } else if refined.inlier_rmse > cfg.max_icp_rmse {
        result.reason = format!("high ICP rmse ({} m)", fixed(refined.inlier_rmse, 3));
    } else if let Some(offset) = Some(height_offset(scan, &refined.transform, &around, cfg)).filter(|&o| ground_disagrees(o, cfg)) {
        result.reason = format!("terrain heights disagree by {} m", signed(offset, 2));
    } else if !good {
        result.reason = why;
    } else {
        result.success = true;
        result.reason = format!("from the prior, height corrected by {} m", signed(dz, 2));
    }
    result.icp = Some(refined);
    result.seconds = start.elapsed().as_secs_f64();
    Ok((result, used))
}

/// State the recovery works on.
struct Survey<'a> {
    scans: &'a [ScanFeatures],
    cfg: &'a CoregConfig,
    graph: Graph,
    results: Vec<PairResult>,
    edge_to_pair: Vec<usize>,
}

impl Survey<'_> {
    fn add_pair_edge(&mut self, k: usize) -> Result<()> {
        let pair = &self.results[k];
        let (fitness, rmse, n) = edge_quality(pair);
        let info = edge_information(pair, self.cfg);
        let (i, j, t) = (pair.i, pair.j, pair.transform);
        self.graph.add_edge(i, j, &t, info, fitness, rmse, n)?;
        self.edge_to_pair.push(k);
        Ok(())
    }

    /// Edges for a scan placed against the combined survey (`_tie_in`): each
    /// neighbour it registers to pairwise from there is an edge; failing all,
    /// the placement itself is one edge to the nearest neighbour.
    fn tie_in(&mut self, k: usize, world_from_scan: &Mat4, neighbours: &[usize], refined: &IcpResult, how: &str) -> Result<()> {
        let (scans, cfg) = (self.scans, self.cfg);
        let mut added = 0;
        for &m in neighbours {
            let initial = matmul(&invert(&self.graph.poses[m]), world_from_scan);
            let mut pair = register_pair(&scans[k], &scans[m], cfg, Some(&initial), None, k as i64, m as i64, None)?;
            if !pair.success {
                continue;
            }
            pair.reason = format!("{how}, then pairwise: {}", pair.reason);
            self.results.push(pair);
            self.add_pair_edge(self.results.len() - 1)?;
            added += 1;
        }
        if added > 0 {
            return Ok(());
        }
        let m = neighbours[0];
        let relative = matmul(&invert(&self.graph.poses[m]), world_from_scan);
        let mut pair = PairResult::new(k as i64, m as i64, &scans[k].name, &scans[m].name);
        pair.transform = relative;
        pair.icp = Some(refined.clone());
        pair.success = true;
        pair.reason = format!("{how} against {} combined scans, fitness {}; no single pair passes", neighbours.len(), fixed(refined.fitness, 3));
        self.results.push(pair);
        self.edge_to_pair.push(self.results.len() - 1);
        // The joint ICP's target frame is the world.
        let info = refined.information.as_ref().map(|i| pg::plane_edge_information(&i.hessian, i.sigma, i.n as i64, world_from_scan, cfg.information_patch_points, cfg.information_min_sigma));
        self.graph.add_edge(k as i64, m as i64, &relative, info, refined.fitness, refined.inlier_rmse, refined.n_correspondences as i64)
    }

    /// Retry failed scans against the combined registered survey
    /// (`_recover_unregistered`); the number recovered.
    fn recover(&mut self, registered: &mut [bool], priors: Option<&[Option<Mat4>]>, log: Log) -> Result<usize> {
        let (scans, cfg) = (self.scans, self.cfg);
        let mut recovered = 0;
        for _ in 0..cfg.recovery_rounds.max(1) {
            let pending: Vec<usize> = (0..scans.len()).filter(|&k| !registered[k] && scans[k].usable()).collect();
            if pending.is_empty() {
                break;
            }
            let combined = combined_stem_map(scans, registered, &self.graph.poses);
            if (combined.len() as i64) < cfg.min_match_inliers && priors.is_none() {
                break;
            }
            log(&format!("Retrying {} unregistered scan(s) against the {}-stem combined survey ...", pending.len(), combined.len()));
            let mut gained = 0;
            for k in pending {
                let name = left(&scans[k].name, 24);
                let mut placed = place_against_survey(scans, k, &combined, registered, &self.graph.poses, cfg)?;
                let mut how = "recovered";
                let prior = priors.and_then(|p| p[k]);
                if let (Some(p), Some(prior)) = (&placed, prior) {
                    let (good, why) = prior_ok(&p.0, &prior, scans[k].origin, cfg);
                    if !good {
                        log(&format!("  {name} placement refused by the prior: {why}"));
                        placed = None;
                    }
                }
                if let (None, Some(prior)) = (&placed, prior) {
                    let reg: Vec<usize> = (0..registered.len()).filter(|&m| registered[m]).collect();
                    let survey: Vec<&ScanFeatures> = reg.iter().map(|&m| &scans[m]).collect();
                    let poses: Vec<Mat4> = reg.iter().map(|&m| self.graph.poses[m]).collect();
                    let (r, used) = place_from_prior(&scans[k], &survey, &poses, &prior, cfg, None)?;
                    if r.success {
                        placed = Some((r.transform, used.iter().map(|&u| reg[u]).collect(), r.icp.expect("placed by ICP")));
                        how = "placed from its prior";
                    } else {
                        log(&format!("  {name} not placed from its prior ({})", r.reason));
                    }
                }
                let Some((world_from_scan, neighbours, refined)) = placed else {
                    log(&format!("  {name} still unplaced"));
                    continue;
                };
                self.graph.poses[k] = world_from_scan;
                self.tie_in(k, &world_from_scan, &neighbours, &refined, how)?;
                registered[k] = true;
                gained += 1;
                log(&format!("  {name} {how} against {} registered scan(s)", neighbours.len()));
            }
            recovered += gained;
            if gained == 0 {
                break;
            }
        }
        Ok(recovered)
    }

    /// Joint multi-view refinement of the registered poses (`_refine_multiview`).
    fn refine_multiview(&mut self, registered: &[bool], reference: usize, log: Log) -> Result<()> {
        let (scans, cfg) = (self.scans, self.cfg);
        let edges: Vec<(usize, usize)> = self.graph.edges.iter().filter(|e| registered[e.i] && registered[e.j]).map(|e| (e.i, e.j)).collect();
        if edges.is_empty() {
            return Ok(());
        }
        let stems: Vec<Vec<Point>> = scans.iter().enumerate().map(|(k, s)| if registered[k] { s.stem_positions() } else { Vec::new() }).collect();
        let points: Vec<Vec<Point>> = scans.iter().enumerate().map(|(k, s)| if registered[k] { s.icp_points.clone() } else { Vec::new() }).collect();
        log(&format!("Joint multi-view refinement: {} scans, {} pairs, {} stems ...", registered.iter().filter(|&&r| r).count(), edges.len(), stems.iter().map(Vec::len).sum::<usize>()));
        let params = crate::coreg::refine::RefineParams {
            voxel_sizes: cfg.refinement_voxel_sizes.clone(),
            max_distances: cfg.refinement_max_distances.clone(),
            rounds: cfg.refinement_rounds.max(1) as usize,
            stem_weight: cfg.refinement_stem_weight,
            stem_radius: cfg.refinement_stem_radius,
            min_voxel_points: cfg.refinement_min_voxel_points,
            points_per_scan: cfg.refinement_points_per_scan,
            ..crate::coreg::refine::RefineParams::default()
        };
        let outcome = crate::coreg::refine::refine_joint(&points, &self.graph.poses, &edges, &stems, reference, &params, &mut |m: &str| log(m))?;
        let moved: Vec<f64> = (0..scans.len()).filter(|&k| registered[k]).map(|k| outcome.shifts[k]).collect();
        let worst = argmax(&outcome.shifts);
        let max_shift = outcome.shifts[worst];
        let max_rotation = outcome.rotations[argmax(&outcome.rotations)];
        log(&format!(
            "  residual {} -> {} cm; shifts median {} cm, max {} cm ({}), rotation max {} deg",
            fixed(outcome.residual_before * 100.0, 2),
            fixed(outcome.residual_after * 100.0, 2),
            fixed(median(&moved) * 100.0, 1),
            fixed(max_shift * 100.0, 1),
            scans[worst].name,
            fixed(max_rotation, 3)
        ));
        if max_shift > cfg.refinement_max_shift {
            log(&format!("  WARNING: {} moved {} cm, more than refinement_max_shift; the pairwise poses are kept", scans[worst].name, fixed(max_shift * 100.0, 0)));
            return Ok(());
        }
        for (k, &ok) in registered.iter().enumerate() {
            if ok && !self.graph.is_fixed(k) {
                self.graph.poses[k] = outcome.poses[k];
            }
        }
        Ok(())
    }
}

/// `np.argmax`: the first largest value, the first NaN if there is one.
fn argmax(v: &[f64]) -> usize {
    if let Some(k) = v.iter().position(|x| x.is_nan()) {
        return k;
    }
    let mut best = 0;
    for (k, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = k;
        }
    }
    best
}

// -------------------------------------------------------------------- survey

/// Pairwise registration and the global solve on prepared scans
/// (`coregister_prepared`).
///
/// `already` is how long the run has taken so far (s), for the reported
/// duration. Pairs between two fixed scans are not tried; with priors, a
/// pair or placement that puts a scanner more than `max_prior_shift` from
/// its prior position is refused, and scans whose stems do not match are
/// placed from their priors.
pub fn coregister_prepared(scans: &[ScanFeatures], cfg: &CoregConfig, opts: &SurveyOptions, log: Log, already: f64) -> Result<SurveyResult> {
    let start = Instant::now();
    let n = scans.len();
    let fixed_nodes: Vec<usize> = opts.fixed.iter().map(|(k, _)| *k).collect();
    let is_fixed = |k: usize| fixed_nodes.contains(&k);
    if let Some(p) = &opts.priors {
        if p.len() < n {
            return Err(Error::invalid("list index out of range"));
        }
    }
    let priors = opts.priors.as_deref();
    let located: Option<Vec<Vec<f64>>> = match (&opts.approximate_positions, priors) {
        (None, Some(priors)) => Some(priors.iter().zip(scans).map(|(p, s)| match p {
            Some(p) => s.location(p).to_vec(),
            None => vec![f64::NAN; 3],
        }).collect()),
        _ => None,
    };
    let positions = opts.approximate_positions.as_deref().or(located.as_deref());

    let usable: Vec<usize> = (0..n).filter(|&k| scans[k].usable()).collect();
    if usable.len() < n {
        let names: Vec<&str> = scans.iter().filter(|s| !s.usable()).map(|s| s.name.as_str()).collect();
        log(&format!("  {} scan(s) set aside: {}", n - usable.len(), names.join(", ")));
    }
    let candidates: Vec<(usize, usize)> = match &opts.pairs {
        Some(p) => {
            if let Some(&(i, j)) = p.iter().find(|&&(i, j)| i >= n || j >= n) {
                return Err(Error::invalid(format!("pair ({i}, {j}) names a scan that does not exist")));
            }
            p.clone()
        }
        None => {
            let mut c = Vec::new();
            for (a, &i) in usable.iter().enumerate() {
                for &j in &usable[a + 1..] {
                    if !(is_fixed(i) && is_fixed(j)) {
                        c.push((i, j));
                    }
                }
            }
            within_reach(&c, positions, cfg.max_pair_distance, log)?
        }
    };
    let workers = resolve_workers(cfg.workers, 0.0, candidates.len());

    let mut results: Vec<PairResult> = Vec::new();
    let matches: Vec<((usize, usize), Option<StemMatch>)> = if cfg.screen_pairs {
        log(&format!("Screening {} pairs by stem matching ...", candidates.len()));
        let (kept, rejected) = screen(&candidates, scans, cfg, workers);
        if rejected.len() <= 20 {
            for p in &rejected {
                log(&format!("  {}", p.summary()));
            }
        }
        log(&format!("  kept {} of {} pairs for ICP ({} rejected by stem screening)", kept.len(), candidates.len(), rejected.len()));
        results.extend(rejected);
        kept.into_iter().map(|(ij, m)| (ij, Some(m))).collect()
    } else {
        candidates.iter().map(|&ij| (ij, None)).collect()
    };
    log(&format!("Refining {} pairs with ICP{}", matches.len(), if workers > 1 { format!(" on {workers} workers ...") } else { " ...".into() }));
    // Pairs sharing a target run together, so each target's ICP pyramid is
    // built once and a small cache is enough however large the survey.
    let mut matches = matches;
    matches.sort_by_key(|((i, j), _)| (*j, *i));
    let targets = TargetCache::new(scans, &cfg.icp, workers + 2);
    let refined = ordered_map(matches.len(), workers, |k| {
        let ((i, j), m) = &matches[k];
        let pyramid = targets.get(*j);
        register_pair(&scans[*i], &scans[*j], cfg, None, m.as_ref(), *i as i64, *j as i64, pyramid.get())
    }, |_, p| {
        if let Ok(p) = p {
            log(&format!("  {}", p.summary()));
        }
    });
    let mut refined: Vec<PairResult> = refined.into_iter().collect::<Result<_>>()?;
    if let Some(priors) = priors {
        for p in refined.iter_mut() {
            let (i, j) = (p.i as usize, p.j as usize);
            if let (true, Some(pi), Some(pj)) = (p.success, priors[i], priors[j]) {
                let (good, why) = prior_ok(&matmul(&pj, &p.transform), &pi, scans[i].origin, cfg);
                if !good {
                    p.success = false;
                    p.reason = format!("refused by the prior: {why}");
                    log(&format!("  {} -> {} refused by the prior: {why}", p.name_i, p.name_j));
                }
            }
        }
    }
    results.extend(refined);
    results.sort_by_key(|p| (p.i, p.j));

    if n == 0 {
        return Err(Error::invalid("a pose graph needs at least one node"));
    }
    let mut reference = cfg.reference_scan.max(0).min(n as i64 - 1) as usize;
    if !opts.fixed.is_empty() {
        if !is_fixed(reference) {
            reference = *fixed_nodes.iter().min().expect("fixed");
        }
    } else if !scans[reference].usable() {
        let replacement = usable.first().copied().unwrap_or(reference);
        if replacement != reference {
            log(&format!("  reference {} is unusable; using {}", scans[reference].name, scans[replacement].name));
        }
        reference = replacement;
    }
    let mut survey = Survey { scans, cfg, graph: Graph::new(n, reference, &opts.fixed)?, results, edge_to_pair: Vec::new() };
    for k in 0..survey.results.len() {
        if survey.results[k].success {
            survey.add_pair_edge(k)?;
        }
    }
    if opts.fixed.is_empty() {
        // A reference no accepted pair touches would leave every other scan
        // "unregistered" even when they registered to each other.
        let rerooted = survey.graph.reference_in_largest_component(reference);
        if rerooted != reference {
            log(&format!("  reference {} has no accepted pair; anchoring on {}, the largest registered block", scans[reference].name, scans[rerooted].name));
            reference = rerooted;
            survey.graph.reference = reference;
        }
    }
    survey.graph.initialise();
    let mut optimisation = None;
    if cfg.optimise_globally && !survey.graph.edges.is_empty() {
        log("Optimising the pose graph ...");
        let o = survey.graph.optimise(cfg.reject_outlier_edges);
        log(&format!("  {}", optimisation_repr(&o)));
        optimisation = Some(o);
    }
    let mut registered = survey.graph.registered();
    if cfg.recover_unregistered && !registered.iter().all(|&r| r) {
        let recovered = survey.recover(&mut registered, priors, log)?;
        if recovered > 0 {
            log(&format!("Recovered {recovered} scan(s); re-optimising ..."));
            let o = survey.graph.optimise(cfg.reject_outlier_edges);
            log(&format!("  {}", optimisation_repr(&o)));
            optimisation = Some(o);
            registered = survey.graph.registered();
        }
    }
    if cfg.refine_multiview && registered.iter().filter(|&&r| r).count() > 1 {
        survey.refine_multiview(&registered, reference, log)?;
    }
    let poses = survey.graph.poses.clone();
    if let Some(o) = optimisation.as_mut() {
        o.poses = poses.clone(); // the Python result shares the graph's final poses
    }
    let result = SurveyResult { pairs: survey.results, poses, reference, optimisation, registered, seconds: already + start.elapsed().as_secs_f64(), edge_to_pair: survey.edge_to_pair };
    if !result.registered.iter().all(|&r| r) {
        let missing: Vec<&str> = (0..n).filter(|&k| !result.registered[k]).map(|k| scans[k].name.as_str()).collect();
        log(&format!("WARNING: {} scan(s) could not be registered: {}", missing.len(), missing.join(", ")));
    }
    Ok(result)
}

/// One scan to prepare within [`coregister`].
#[derive(Debug, Clone)]
pub struct ScanSpec {
    pub input: ScanInput,
    pub reflectors: Vec<Reflector>,
    pub levelling: Option<Mat4>,
}

/// Prepare every scan, then coregister them (`coregister`).
///
/// Scans are prepared on several threads when all are files; a scan that
/// cannot be read or prepared is set aside with the reason, not raised.
/// Names default to `scan_00`, `scan_01`, ...
pub fn coregister(specs: &[ScanSpec], cfg: &CoregConfig, names: Option<&[String]>, opts: &SurveyOptions, log: Log) -> Result<(Vec<ScanFeatures>, SurveyResult)> {
    let start = Instant::now();
    let n = specs.len();
    if let Some(names) = names.filter(|v| !v.is_empty()) {
        if names.len() < n {
            return Err(Error::invalid("list index out of range"));
        }
    }
    let scan_names: Vec<String> = (0..n)
        .map(|k| {
            let given = names.filter(|v| !v.is_empty()).map(|v| v[k].clone()).unwrap_or_default();
            if given.is_empty() { format!("scan_{k:02}") } else { given }
        })
        .collect();
    let paths_only = specs.iter().all(|s| matches!(s.input, ScanInput::Path(_)));
    let workers = if paths_only { resolve_workers(cfg.workers, cfg.memory_per_worker_gb, n) } else { 1 };
    log(&format!("Preparing {n} scans{}", if workers > 1 { format!(" on {workers} workers ...") } else { " ...".into() }));
    let scans = ordered_map(n, workers, |k| {
        let t0 = Instant::now();
        let spec = &specs[k];
        prepare_scan(&spec.input, cfg, &scan_names[k], &spec.reflectors, spec.levelling.as_ref(), None).unwrap_or_else(|e| {
            let source = match &spec.input {
                ScanInput::Path(p) => Some(p.to_string_lossy().into_owned()),
                _ => None,
            };
            unusable_scan(&scan_names[k], 0, source, t0, &e.describe())
        })
    }, |_, f| {
        if f.error.is_empty() {
            log(&format!("  {} {} pts -> {:3} stems, {} ICP points ({} s)", left(&f.name, 24), crate::util::pyformat::right(&thousands(f.n_points as i64), 10), f.stems.len(), crate::util::pyformat::right(&thousands(f.icp_points.len() as i64), 8), fixed(f.seconds, 1)));
        } else {
            log(&format!("  {} SET ASIDE: {}", left(&f.name, 24), f.error));
        }
    });
    let survey = coregister_prepared(&scans, cfg, opts, log, 0.0).map(|mut r| {
        r.seconds = start.elapsed().as_secs_f64();
        r
    })?;
    Ok((scans, survey))
}

// ------------------------------------------------------------------ results

/// What the report and the saved file need of a scan.
#[derive(Debug, Clone)]
pub struct ScanSummary {
    pub name: String,
    pub n_points: i64,
    pub n_stems: i64,
    pub error: String,
    pub source: Option<String>,
    pub levelling: Mat4,
}

impl From<&ScanFeatures> for ScanSummary {
    fn from(s: &ScanFeatures) -> Self {
        ScanSummary { name: s.name.clone(), n_points: s.n_points as i64, n_stems: s.stems.len() as i64, error: s.error.clone(), source: s.source.clone(), levelling: s.levelling }
    }
}

/// `world_from_scan` of one scan, applying directly to its raw points.
pub fn transform_for(pose: &Mat4, levelling: &Mat4) -> Mat4 {
    matmul(pose, levelling)
}

/// Horizontal stem disagreement per accepted pair under `poses`
/// (`SurveyResult.consistency`): the median (or, not `robust`, the RMS)
/// tree-to-tree distance, keyed as a Python dict (a repeated pair keeps its
/// first place and its last value).
pub fn consistency(pairs: &[PairResult], poses: &[Mat4], robust: bool) -> Result<Vec<((i64, i64), f64)>> {
    let mut out: Vec<((i64, i64), f64)> = Vec::new();
    for p in pairs {
        if !p.success || p.matched_source.is_empty() {
            continue;
        }
        let pose = |k: i64| {
            let k = if k < 0 { k + poses.len() as i64 } else { k };
            usize::try_from(k).ok().and_then(|k| poses.get(k)).copied().ok_or_else(|| Error::invalid("list index out of range"))
        };
        let relative = matmul(&invert(&pose(p.j)?), &pose(p.i)?);
        let moved = transform_points(&relative, &p.matched_source);
        let d: Vec<f64> = moved
            .iter()
            .zip(&p.matched_target)
            .map(|(a, b)| {
                let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        let value = if robust { median(&d) } else { (pairwise_sum(&d.iter().map(|x| x * x).collect::<Vec<_>>()) / d.len() as f64).sqrt() };
        match out.iter_mut().find(|(k, _)| *k == (p.i, p.j)) {
            Some(entry) => entry.1 = value,
            None => out.push(((p.i, p.j), value)),
        }
    }
    Ok(out)
}

/// The pairs behind the edges the global solve rejected.
pub fn rejected_pairs<'a>(pairs: &'a [PairResult], edge_to_pair: &[usize], optimisation: Option<&pg::Optimisation>) -> Vec<&'a PairResult> {
    let Some(o) = optimisation else { return Vec::new() };
    o.rejected_edges.iter().filter_map(|&k| edge_to_pair.get(k).and_then(|&p| pairs.get(p))).collect()
}

fn name_at(scans: &[ScanSummary], k: i64) -> Result<&str> {
    let n = scans.len() as i64;
    let k = if k < 0 { k + n } else { k };
    usize::try_from(k).ok().and_then(|k| scans.get(k)).map(|s| s.name.as_str()).ok_or_else(|| Error::invalid("list index out of range"))
}

/// Plain-text summary for a log or a QC record (`SurveyResult.report`).
#[allow(clippy::too_many_arguments)]
pub fn report(scans: &[ScanSummary], pairs: &[PairResult], poses: &[Mat4], reference: i64, optimisation: Option<&pg::Optimisation>, registered: &[bool], seconds: f64, edge_to_pair: &[usize]) -> Result<String> {
    let mut lines = vec![
        format!("Coregistration of {} scans ({} registered) in {} s", scans.len(), registered.iter().filter(|&&r| r).count(), fixed(seconds, 1)),
        format!("reference scan: {}", name_at(scans, reference)?),
        String::new(),
        "Scans:".to_string(),
    ];
    for (k, s) in scans.iter().enumerate() {
        let flag = if !s.error.is_empty() {
            format!("SET ASIDE ({})", s.error)
        } else if *registered.get(k).ok_or_else(|| Error::invalid("list index out of range"))? {
            "registered".into()
        } else {
            "NOT REGISTERED".into()
        };
        lines.push(format!("  {k:2} {} {} pts  {:3} stems  {flag}", left(&s.name, 24), crate::util::pyformat::right(&thousands(s.n_points), 10), s.n_stems));
    }
    lines.push(String::new());
    lines.push("Pairs:".into());
    lines.extend(pairs.iter().map(|p| format!("  {}", p.summary())));
    let c = consistency(pairs, poses, true)?;
    if !c.is_empty() {
        lines.push(String::new());
        lines.push("Stem agreement under the final poses (median horizontal tree-to-tree distance, no ground truth needed):".into());
        // Flag pairs against the survey itself: the absolute level depends on
        // the stand and the spacing, but an outlier is always worth a look.
        let m = 3.0 * median(&c.iter().map(|x| x.1).collect::<Vec<_>>());
        let threshold = if 0.10 > m { 0.10 } else { m };
        let mut sorted = c.clone();
        sorted.sort_by_key(|x| x.0);
        for ((i, j), value) in sorted {
            let flag = if value <= threshold { "" } else { "   <-- check" };
            lines.push(format!("  {} <-> {}: {} cm{flag}", name_at(scans, i)?, name_at(scans, j)?, fixed_width(value * 100.0, 6, 2)));
        }
    }
    if let Some(o) = optimisation {
        lines.push(String::new());
        lines.push(format!("Global optimisation: {} iterations, converged={}, error {} -> {}", o.iterations, boolean(o.converged), general(o.initial_error, 4), general(o.final_error, 4)));
        if !o.rejected_edges.is_empty() {
            let names: Vec<String> = rejected_pairs(pairs, edge_to_pair, Some(o)).iter().map(|p| format!("{}->{}", p.name_i, p.name_j)).collect();
            lines.push(format!("  rejected inconsistent pairs: {}", names.join(", ")));
        }
    }
    Ok(lines.join("\n"))
}

fn json_number(x: f64) -> Json {
    if x.is_finite() {
        Json::Float(x)
    } else {
        Json::Null
    }
}

fn json_matrix(t: &Mat4) -> Json {
    Json::Array((0..4).map(|r| Json::Array((0..4).map(|c| Json::Float(t[(r, c)])).collect())).collect())
}

/// The text of `transforms.json` (`SurveyResult.save`): the transforms and
/// quality of every scan and pair, laid out as `json.dumps(indent=2)`.
pub fn survey_json(scans: &[ScanSummary], pairs: &[PairResult], poses: &[Mat4], reference: i64, registered: &[bool], seconds: f64) -> Result<String> {
    if poses.len() < scans.len() || registered.len() < scans.len() {
        return Err(Error::invalid("list index out of range"));
    }
    let scans_json = scans
        .iter()
        .enumerate()
        .map(|(k, s)| {
            Json::Object(vec![
                ("index".into(), Json::Int(k as i64)),
                ("name".into(), Json::Str(s.name.clone())),
                ("source".into(), s.source.as_ref().map_or(Json::Null, |p| Json::Str(p.clone()))),
                ("n_points".into(), Json::Int(s.n_points)),
                ("n_stems".into(), Json::Int(s.n_stems)),
                ("registered".into(), Json::Bool(registered[k])),
                ("world_from_scan".into(), json_matrix(&transform_for(&poses[k], &s.levelling))),
                ("levelling".into(), json_matrix(&s.levelling)),
            ])
        })
        .collect();
    let pairs_json = pairs
        .iter()
        .map(|p| {
            Json::Object(vec![
                ("i".into(), Json::Int(p.i)),
                ("j".into(), Json::Int(p.j)),
                ("name_i".into(), Json::Str(p.name_i.clone())),
                ("name_j".into(), Json::Str(p.name_j.clone())),
                ("success".into(), Json::Bool(p.success)),
                ("reason".into(), Json::Str(p.reason.clone())),
                ("stem_matches".into(), Json::Int(p.n_stem_matches() as i64)),
                ("fitness".into(), json_number(p.fitness())),
                ("rmse".into(), json_number(p.rmse())),
                ("n_correspondences".into(), Json::Int(p.icp.as_ref().map_or(0, |r| r.n_correspondences as i64))),
                ("coarse_stem_rmse".into(), json_number(p.coarse_stem_rmse)),
                ("fine_stem_rmse".into(), json_number(p.fine_stem_rmse)),
                ("used_icp".into(), Json::Bool(p.used_icp)),
                ("trusted".into(), Json::Bool(p.trusted)),
                ("transform".into(), json_matrix(&p.transform)),
            ])
        })
        .collect();
    let payload = Json::Object(vec![("reference".into(), Json::Int(reference)), ("seconds".into(), Json::Float(seconds)), ("scans".into(), Json::Array(scans_json)), ("pairs".into(), Json::Array(pairs_json))]);
    Ok(json::to_string_indented(&payload, 2))
}

/// Write `transforms.json`, creating its directory.
pub fn save_survey(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// The `world_from_scan` of every registered scan in a `transforms.json`
/// (`load_transforms`), in the order of a Python dict built from them.
pub fn load_transforms(path: &Path) -> Result<Vec<(String, Vec<Vec<f64>>)>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::file(path, e.to_string()))?;
    let payload = json::parse(&text)?;
    let Some(Json::Array(scans)) = payload.get("scans") else { return Err(Error::invalid("a transforms file needs a `scans` list")) };
    let number = |v: &Json| -> Result<f64> {
        match v {
            Json::Int(i) => Ok(*i as f64),
            Json::Float(f) => Ok(*f),
            Json::Bool(b) => Ok(*b as i64 as f64),
            _ => Err(Error::invalid("`world_from_scan` must hold numbers")),
        }
    };
    let mut out: Vec<(String, Vec<Vec<f64>>)> = Vec::new();
    for s in scans {
        if !s.get("registered").is_some_and(Json::truthy) {
            continue;
        }
        let Some(Json::Str(name)) = s.get("name") else { return Err(Error::invalid("a scan entry needs a `name`")) };
        let Some(Json::Array(rows)) = s.get("world_from_scan") else { return Err(Error::invalid("a scan entry needs `world_from_scan`")) };
        let matrix = rows
            .iter()
            .map(|r| match r {
                Json::Array(v) => v.iter().map(number).collect::<Result<Vec<f64>>>(),
                _ => Err(Error::invalid("`world_from_scan` must be a list of rows")),
            })
            .collect::<Result<Vec<_>>>()?;
        match out.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = matrix,
            None => out.push((name.clone(), matrix)),
        }
    }
    Ok(out)
}

// -------------------------------------------------------------------- merge

/// Every scan moved into the reference frame and concatenated
/// (`merge_clouds`), with a `scan_id` attribute.
///
/// A scan is moved by `poses[k] @ levellings[k]`; unregistered scans are
/// left out if `only_registered`. With `voxel`, each scan and then the
/// merged cloud are thinned to one point per voxel (the first).
pub fn merge_clouds(inputs: &[ScanInput], poses: &[Mat4], levellings: &[Mat4], registered: &[bool], only_registered: bool, voxel: Option<f64>, cfg: &CoregConfig) -> std::result::Result<PointCloud, PrepareError> {
    let voxel = voxel.filter(|&v| v != 0.0);
    if let Some(v) = voxel {
        if v.is_nan() || v <= 0.0 {
            return Err(Error::invalid("voxel size must be positive").into());
        }
    }
    let mut xyz: Vec<Point> = Vec::new();
    let mut ids: Vec<i32> = Vec::new();
    for (k, input) in inputs.iter().enumerate() {
        let out_of_range = || PrepareError::Core(Error::invalid("list index out of range"));
        if only_registered && !*registered.get(k).ok_or_else(out_of_range)? {
            continue;
        }
        let points = match input {
            ScanInput::Path(p) => read_scan(p, cfg)?,
            ScanInput::Points(p) => p.clone(),
            ScanInput::Failed { reason, .. } => return Err(Error::invalid(reason.clone()).into()),
        };
        let t = transform_for(poses.get(k).ok_or_else(out_of_range)?, levellings.get(k).ok_or_else(out_of_range)?);
        let mut moved = transform_points(&t, &points);
        if let Some(v) = voxel {
            moved = crate::coreg::geometry::voxel_centroids(&moved, v, false).0;
        }
        ids.extend(std::iter::repeat_n(k as i32, moved.len()));
        xyz.extend(moved);
    }
    if let Some(v) = voxel {
        let keep = crate::filters::voxel_downsample_indices(&xyz, v);
        xyz = keep.iter().map(|&i| xyz[i]).collect();
        ids = keep.iter().map(|&i| ids[i]).collect();
    }
    let mut cloud = PointCloud::new(xyz);
    cloud.set_attr("scan_id", Attr::I32(ids))?;
    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_follow_the_request() {
        assert_eq!(resolve_workers(4, 4.0, 10), 4);
        assert_eq!(resolve_workers(4, 4.0, 2), 2);
        assert_eq!(resolve_workers(0, 0.0, 1), 1);
        assert!(resolve_workers(0, 0.0, 1000) >= 1);
    }

    #[test]
    fn pairs_within_reach() {
        let pos = vec![vec![0.0, 0.0, 0.0], vec![30.0, 0.0, 0.0], vec![f64::NAN, 1.0, 0.0]];
        let seen = std::sync::Mutex::new(Vec::new());
        let log = |m: &str| seen.lock().unwrap().push(m.to_string());
        let kept = within_reach(&[(0, 1), (0, 2), (1, 2)], Some(&pos), 10.0, &log).unwrap();
        assert_eq!(kept, vec![(0, 2), (1, 2)]);
        assert_eq!(seen.lock().unwrap().as_slice(), ["  1 of 3 pairs skipped: more than 10 m apart"]);
        assert_eq!(within_reach(&[(0, 1)], None, 10.0, &log).unwrap(), vec![(0, 1)]);
        assert_eq!(within_reach(&[(0, 1)], Some(&pos), f64::INFINITY, &log).unwrap(), vec![(0, 1)]);
    }

    #[test]
    fn optimisation_reads_as_python() {
        let o = pg::Optimisation { poses: Vec::new(), iterations: 30, converged: true, initial_error: 2537.0, final_error: 4.613e-12, rejected_edges: vec![1], edge_errors: Vec::new() };
        assert_eq!(optimisation_repr(&o), "OptimisationResult(iterations=30, converged=True, error 2537 -> 4.613e-12, rejected=1)");
    }

    #[test]
    fn argmax_as_numpy() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[1.0, f64::NAN, 5.0]), 1);
    }

    #[test]
    fn transforms_round_trip() {
        let dir = std::env::temp_dir().join(format!("sylva-survey-{}", std::process::id()));
        let scans = vec![ScanSummary { name: "a\u{e9}".into(), n_points: 12, n_stems: 3, error: String::new(), source: None, levelling: Matrix4::identity() }, ScanSummary { name: "b".into(), n_points: 5, n_stems: 0, error: "bad".into(), source: Some("/x/b.laz".into()), levelling: Matrix4::identity() }];
        let poses = vec![Matrix4::identity(), crate::coreg::transforms::yaw_transform(0.3, 1.0, 2.0, 3.0)];
        let text = survey_json(&scans, &[], &poses, 0, &[true, true], 1.5).unwrap();
        assert!(text.contains("\"name\": \"a\\u00e9\"") && text.contains("\"seconds\": 1.5,"));
        let path = dir.join("sub").join("t.json");
        save_survey(&path, &text).unwrap();
        let loaded = load_transforms(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1].1[0][3], 1.0);
        std::fs::remove_dir_all(dir).ok();
    }
}
