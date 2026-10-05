// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Tree architecture from a cylinder model.
//!
//! A branch is one `branch_id` chain of cylinders: it starts at the cylinder
//! whose parent belongs to another chain and follows its own children to the
//! tip. Branch angles, lengths and radii, the stem's taper, lean and sweep,
//! the crown outlined by the branches, and how much of the model was fitted
//! to points rather than filled in by priors.

use std::collections::HashMap;

use super::model::{Cylinder, Qsm};
use crate::transform::{add, dot, norm, scale, sub};
use crate::trees::{crown_shape, CrownShape};
use crate::Point;

/// Distance along a branch (m) over which its insertion direction is taken.
pub const INSERTION_REACH: f64 = 0.5;

/// One branch (one `branch_id` chain); the stem is the order-0 chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub id: u32,
    pub order: u32,
    /// Branch the first cylinder hangs from (-1 for the stem).
    pub parent: i64,
    pub n_cylinders: usize,
    /// Sum of cylinder lengths (m) and volume (m3).
    pub length: f64,
    pub volume: f64,
    /// Radius of the first cylinder and length-weighted mean radius (m).
    pub base_radius: f64,
    pub mean_radius: f64,
    /// Heights of the branch base and tip above the tree base (m).
    pub base_height: f64,
    pub tip_height: f64,
    /// Angle (deg) between the branch's direction over 0.5 m after its first
    /// cylinder (which only joins it to the parent's axis) and the parent's
    /// direction over 0.5 m either side of the junction (NaN for the stem).
    pub insertion_angle: f64,
    /// Chord from base to tip: angle from the vertical (deg, 0 = straight up)
    /// and direction (deg, counter-clockwise from +x).
    pub zenith: f64,
    pub azimuth: f64,
    /// Length over chord length (1 = straight).
    pub tortuosity: f64,
    /// Branches growing from this one.
    pub n_children: usize,
    /// Share of the length whose radius was fitted to points.
    pub measured_fraction: f64,
}

/// Whole-tree metrics of a QSM.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeMetrics {
    pub height: f64,
    pub dbh: f64,
    pub total_volume: f64,
    pub stem_volume: f64,
    pub branch_volume: f64,
    pub total_length: f64,
    pub stem_length: f64,
    pub max_order: u32,
    /// Per branch order 0, 1, 2, ...: number of branches, length (m), volume (m3).
    pub n_branches_by_order: Vec<usize>,
    pub length_by_order: Vec<f64>,
    pub volume_by_order: Vec<f64>,
    pub n_tips: usize,
    /// Mean over tips of the path length from the base, over the longest
    /// such path (Smith et al. 2014); 1 for a single stem.
    pub path_fraction: f64,
    /// Height of the lowest first-order branch at least `crown_branch_length` long.
    pub crown_base_height: f64,
    /// Stem lean (deg from vertical, chord from the base to the stem at
    /// `min(5 m, height / 2)`) and its direction (deg).
    pub lean: f64,
    pub lean_direction: f64,
    /// Greatest distance of the stem axis from the chord between the base
    /// and the crown base (or the stem tip), over the chord length.
    pub sweep: f64,
    /// Stem radius profile: heights above the base (m) and radii (m).
    pub taper_heights: Vec<f64>,
    pub taper_radii: Vec<f64>,
    /// Crown outlined by the branches (points along every cylinder above the crown base).
    pub crown: CrownShape,
    /// Share of the volume and of the length in cylinders fitted to points
    /// (the rest came from taper and pipe-model priors).
    pub measured_volume_fraction: f64,
    pub measured_length_fraction: f64,
    /// Length-weighted medians over first-order branches (deg).
    pub median_insertion_angle: f64,
    pub median_branch_zenith: f64,
}

fn angle_deg(a: &Point, b: &Point) -> f64 {
    let c = dot(a, b) / (norm(a) * norm(b)).max(1e-18);
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Direction of a chain over `INSERTION_REACH` metres either side of the end
/// of cylinder `i`.
fn local_direction(cyl: &[Cylinder], position: &[(usize, usize)], chains: &[&Vec<usize>], i: usize) -> Point {
    let (k, j) = position[i];
    let order = chains[k];
    let mut cum = vec![0.0; order.len() + 1];
    for (m, &c) in order.iter().enumerate() {
        cum[m + 1] = cum[m] + cyl[c].length;
    }
    let at = |s: f64| -> Point {
        let s = s.clamp(0.0, cum[order.len()]);
        let m = (0..order.len()).find(|&m| cum[m + 1] >= s).unwrap_or(order.len() - 1);
        let c = &cyl[order[m]];
        add(&c.start, &scale(&c.axis, (s - cum[m]).clamp(0.0, c.length)))
    };
    let s = cum[j + 1];
    sub(&at(s + INSERTION_REACH), &at(s - INSERTION_REACH))
}

fn weighted_median(values: &[(f64, f64)]) -> f64 {
    let mut v: Vec<(f64, f64)> = values.iter().copied().filter(|(x, w)| x.is_finite() && *w > 0.0).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    for (x, w) in &v {
        acc += w;
        if acc >= total / 2.0 {
            return *x;
        }
    }
    v[v.len() - 1].0
}

/// Cylinders of each branch in order from base to tip.
pub(crate) fn chains(cyl: &[Cylinder]) -> Vec<(u32, Vec<usize>)> {
    let mut by_branch: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c) in cyl.iter().enumerate() {
        by_branch.entry(c.branch_id).or_default().push(i);
    }
    let mut out: Vec<(u32, Vec<usize>)> = Vec::with_capacity(by_branch.len());
    for (id, members) in by_branch {
        let first = members.iter().copied().find(|&i| {
            let p = cyl[i].parent;
            p < 0 || cyl[p as usize].branch_id != id
        });
        let Some(mut cur) = first else { continue };
        let mut next_of: HashMap<usize, usize> = HashMap::new();
        for &i in &members {
            let p = cyl[i].parent;
            if p >= 0 && cyl[p as usize].branch_id == id {
                // The longest continuation wins where a chain splits (it should not).
                next_of.entry(p as usize).or_insert(i);
            }
        }
        let mut order = vec![cur];
        while let Some(&n) = next_of.get(&cur) {
            if order.len() > members.len() {
                break;
            }
            order.push(n);
            cur = n;
        }
        out.push((id, order));
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

/// Per-branch table of a QSM.
pub fn branches(qsm: &Qsm) -> Vec<Branch> {
    let cyl = &qsm.cylinders;
    if cyl.is_empty() {
        return Vec::new();
    }
    let base_z = cyl.iter().filter(|c| c.parent < 0).map(|c| c.start[2]).fold(f64::INFINITY, f64::min);
    let chains = chains(cyl);
    // Position of every cylinder in its chain, for directions along a chain.
    let mut position = vec![(0usize, 0usize); cyl.len()];
    let chain_of: Vec<&Vec<usize>> = chains.iter().map(|(_, c)| c).collect();
    for (k, (_, order)) in chains.iter().enumerate() {
        for (j, &i) in order.iter().enumerate() {
            position[i] = (k, j);
        }
    }
    let mut children: HashMap<u32, usize> = HashMap::new();
    for (id, order) in &chains {
        let p = cyl[order[0]].parent;
        if p >= 0 {
            let pb = cyl[p as usize].branch_id;
            if pb != *id {
                *children.entry(pb).or_default() += 1;
            }
        }
    }
    chains
        .iter()
        .map(|(id, order)| {
            let first = &cyl[order[0]];
            let last = &cyl[*order.last().unwrap()];
            let length: f64 = order.iter().map(|&i| cyl[i].length).sum();
            let volume: f64 = order.iter().map(|&i| cyl[i].volume()).sum();
            let mean_radius = if length > 0.0 { order.iter().map(|&i| cyl[i].radius * cyl[i].length).sum::<f64>() / length } else { first.radius };
            let measured: f64 = order.iter().filter(|&&i| cyl[i].n_points > 0).map(|&i| cyl[i].length).sum();
            let tip = last.end();
            let chord = sub(&tip, &first.start);
            let chord_len = norm(&chord);
            let parent = if first.parent >= 0 { cyl[first.parent as usize].branch_id as i64 } else { -1 };
            // Branch direction over `INSERTION_REACH` metres after its first
            // cylinder: that one starts on the parent's axis, so it records the
            // junction rather than the branch.
            let (from, skip) = if order.len() > 1 { (first.end(), first.length) } else { (first.start, 0.0) };
            let reach = INSERTION_REACH.min((length - skip).max(0.0));
            let mut walked = 0.0;
            let mut far = last.end();
            for &i in order.iter().skip(usize::from(order.len() > 1)) {
                let c = &cyl[i];
                if walked + c.length >= reach {
                    far = add(&c.start, &scale(&c.axis, (reach - walked).max(0.0)));
                    break;
                }
                walked += c.length;
                far = c.end();
            }
            let dir = sub(&far, &from);
            // Parent direction over `INSERTION_REACH` either side of the
            // junction: the parent cylinder at a fork leans towards the branch.
            let parent_dir = if first.parent >= 0 { local_direction(cyl, &position, &chain_of, first.parent as usize) } else { [0.0, 0.0, 0.0] };
            let insertion = if first.parent >= 0 && parent != *id as i64 && norm(&dir) > 0.0 && norm(&parent_dir) > 0.0 { angle_deg(&dir, &parent_dir) } else { f64::NAN };
            Branch {
                id: *id,
                order: first.branch_order,
                parent: if parent == *id as i64 { -1 } else { parent },
                n_cylinders: order.len(),
                length,
                volume,
                base_radius: first.radius,
                mean_radius,
                base_height: first.start[2] - base_z,
                tip_height: tip[2] - base_z,
                insertion_angle: insertion,
                zenith: if chord_len > 0.0 { (chord[2] / chord_len).clamp(-1.0, 1.0).acos().to_degrees() } else { f64::NAN },
                azimuth: chord[1].atan2(chord[0]).to_degrees(),
                tortuosity: if chord_len > 0.0 { length / chord_len } else { f64::NAN },
                n_children: children.get(id).copied().unwrap_or(0),
                measured_fraction: if length > 0.0 { measured / length } else { 0.0 },
            }
        })
        .collect()
}

/// Whole-tree metrics. `crown_branch_length` (m) is the shortest first-order
/// branch that marks the crown base; `crown_slice` (m) the slice height of the
/// crown volume.
pub fn tree_metrics(qsm: &Qsm, crown_branch_length: f64, crown_slice: f64) -> TreeMetrics {
    let cyl = &qsm.cylinders;
    let br = branches(qsm);
    let max_order = qsm.max_branch_order();
    let n_ord = max_order as usize + 1;
    let mut n_by = vec![0usize; n_ord];
    let mut len_by = vec![0.0; n_ord];
    let mut vol_by = vec![0.0; n_ord];
    for b in &br {
        n_by[b.order as usize] += 1;
        len_by[b.order as usize] += b.length;
        vol_by[b.order as usize] += b.volume;
    }
    let base_z = cyl.iter().filter(|c| c.parent < 0).map(|c| c.start[2]).fold(f64::INFINITY, f64::min);
    let top_z = cyl.iter().map(|c| c.start[2].max(c.end()[2])).fold(f64::NEG_INFINITY, f64::max);
    let total_volume = qsm.total_volume();
    let total_length = qsm.total_length();

    // Path lengths from the base; tips are cylinders without children.
    let mut has_child = vec![false; cyl.len()];
    for c in cyl {
        if c.parent >= 0 {
            has_child[c.parent as usize] = true;
        }
    }
    let mut path = vec![f64::NAN; cyl.len()];
    fn path_to(i: usize, cyl: &[Cylinder], path: &mut [f64]) -> f64 {
        let mut stack = vec![i];
        while let Some(&j) = stack.last() {
            if path[j].is_finite() {
                stack.pop();
                continue;
            }
            let p = cyl[j].parent;
            if p < 0 {
                path[j] = cyl[j].length;
                stack.pop();
            } else if path[p as usize].is_finite() {
                path[j] = path[p as usize] + cyl[j].length;
                stack.pop();
            } else {
                stack.push(p as usize);
            }
        }
        path[i]
    }
    let tips: Vec<usize> = (0..cyl.len()).filter(|&i| !has_child[i]).collect();
    let tip_paths: Vec<f64> = tips.iter().map(|&i| path_to(i, cyl, &mut path)).collect();
    let longest = tip_paths.iter().copied().fold(0.0, f64::max);
    let path_fraction = if longest > 0.0 { tip_paths.iter().sum::<f64>() / tip_paths.len() as f64 / longest } else { f64::NAN };

    // Stem.
    let stem = br.iter().find(|b| b.order == 0);
    let stem_chain: Vec<usize> = chains(cyl).into_iter().find(|(id, _)| stem.is_some_and(|s| s.id == *id)).map(|(_, c)| c).unwrap_or_default();
    let taper_heights: Vec<f64> = stem_chain.iter().map(|&i| cyl[i].start[2] + 0.5 * cyl[i].axis[2] * cyl[i].length - base_z).collect();
    let taper_radii: Vec<f64> = stem_chain.iter().map(|&i| cyl[i].radius).collect();
    let crown_base = br.iter().filter(|b| b.order == 1 && b.length >= crown_branch_length).map(|b| b.base_height).fold(f64::INFINITY, f64::min);
    let crown_base = if crown_base.is_finite() { crown_base } else { f64::NAN };
    let stem_point_at = |h: f64| -> Option<Point> {
        for &i in &stem_chain {
            let c = &cyl[i];
            let (z0, z1) = (c.start[2] - base_z, c.end()[2] - base_z);
            if (z0 <= h && h <= z1) || (z1 <= h && h <= z0) {
                let t = if (z1 - z0).abs() > 1e-12 { (h - z0) / (z1 - z0) } else { 0.0 };
                return Some(add(&c.start, &scale(&c.axis, t * c.length)));
            }
        }
        None
    };
    let height = top_z - base_z;
    let root = stem_chain.first().map(|&i| cyl[i].start);
    let (lean, lean_direction) = match (root, stem_point_at((height / 2.0).min(5.0))) {
        (Some(r), Some(q)) => {
            let d = sub(&q, &r);
            (angle_deg(&d, &[0.0, 0.0, 1.0]), d[1].atan2(d[0]).to_degrees())
        }
        _ => (f64::NAN, f64::NAN),
    };
    let chord_top_h = if crown_base.is_finite() && crown_base > 1.0 { crown_base } else { stem_chain.last().map(|&i| cyl[i].end()[2] - base_z).unwrap_or(f64::NAN) };
    let sweep = match (root, stem_point_at(chord_top_h)) {
        (Some(r), Some(q)) => {
            let d = sub(&q, &r);
            let l = norm(&d);
            if l > 0.0 {
                let u = scale(&d, 1.0 / l);
                let mut worst = 0.0f64;
                for &i in &stem_chain {
                    for p in [cyl[i].start, cyl[i].end()] {
                        let t = dot(&sub(&p, &r), &u);
                        if (0.0..=l).contains(&t) {
                            worst = worst.max(norm(&sub(&sub(&p, &r), &scale(&u, t))));
                        }
                    }
                }
                worst / l
            } else {
                f64::NAN
            }
        }
        _ => f64::NAN,
    };

    // Crown outlined by the branches: points every 10 cm along each cylinder
    // above the crown base (branches only, not the stem inside the crown).
    let z_crown = base_z + if crown_base.is_finite() { crown_base } else { 0.0 };
    let mut outline: Vec<Point> = Vec::new();
    for c in cyl.iter().filter(|c| c.branch_order >= 1) {
        let n = (c.length / 0.1).ceil().max(1.0) as usize;
        for s in 0..=n {
            outline.push(add(&c.start, &scale(&c.axis, c.length * s as f64 / n as f64)));
        }
    }
    let stem_xy = root.map(|r| [r[0], r[1]]);
    let mut crown = crown_shape(&outline, stem_xy, z_crown, crown_slice);
    crown.base_height -= base_z;
    crown.top_height -= base_z;

    let measured_volume: f64 = cyl.iter().filter(|c| c.n_points > 0).map(Cylinder::volume).sum();
    let measured_length: f64 = cyl.iter().filter(|c| c.n_points > 0).map(|c| c.length).sum();
    let first: Vec<&Branch> = br.iter().filter(|b| b.order == 1).collect();
    TreeMetrics {
        height,
        dbh: qsm.dbh(),
        total_volume,
        stem_volume: qsm.stem_volume(),
        branch_volume: qsm.branch_volume(),
        total_length,
        stem_length: stem.map_or(0.0, |s| s.length),
        max_order,
        n_branches_by_order: n_by,
        length_by_order: len_by,
        volume_by_order: vol_by,
        n_tips: tips.len(),
        path_fraction,
        crown_base_height: crown_base,
        lean,
        lean_direction,
        sweep,
        taper_heights,
        taper_radii,
        crown,
        measured_volume_fraction: if total_volume > 0.0 { measured_volume / total_volume } else { 0.0 },
        measured_length_fraction: if total_length > 0.0 { measured_length / total_length } else { 0.0 },
        median_insertion_angle: weighted_median(&first.iter().map(|b| (b.insertion_angle, b.length)).collect::<Vec<_>>()),
        median_branch_zenith: weighted_median(&first.iter().map(|b| (b.zenith, b.length)).collect::<Vec<_>>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyl(start: Point, axis: Point, length: f64, radius: f64, parent: i64, order: u32, branch: u32) -> Cylinder {
        let n = norm(&axis);
        Cylinder { start, axis: scale(&axis, 1.0 / n), length, radius, parent, branch_order: order, branch_id: branch, n_points: 10 }
    }

    #[test]
    fn stem_with_one_branch() {
        // 10 m vertical stem in 1 m cylinders; a 2 m branch at 45 deg from 6 m.
        let mut c: Vec<Cylinder> = (0..10).map(|k| cyl([0.0, 0.0, k as f64], [0.0, 0.0, 1.0], 1.0, 0.2 - 0.01 * k as f64, k as i64 - 1, 0, 0)).collect();
        c.push(cyl([0.0, 0.0, 6.0], [1.0, 0.0, 1.0], 1.0, 0.05, 5, 1, 1));
        c.push(cyl(add(&[0.0, 0.0, 6.0], &scale(&[1.0, 0.0, 1.0], 1.0 / 2f64.sqrt())), [1.0, 0.0, 1.0], 1.0, 0.04, 10, 1, 1));
        let q = Qsm { cylinders: c };
        let b = branches(&q);
        assert_eq!(b.len(), 2);
        let br = b.iter().find(|x| x.order == 1).unwrap();
        assert!((br.insertion_angle - 45.0).abs() < 1e-6 && (br.zenith - 45.0).abs() < 1e-6);
        assert!((br.length - 2.0).abs() < 1e-9 && (br.tortuosity - 1.0).abs() < 1e-9);
        assert!((br.base_height - 6.0).abs() < 1e-9 && br.parent == 0);
        let m = tree_metrics(&q, 1.0, 0.5);
        assert!((m.height - 10.0).abs() < 1e-9);
        assert!((m.crown_base_height - 6.0).abs() < 1e-9);
        assert!(m.lean.abs() < 1e-9 && m.sweep.abs() < 1e-9);
        assert_eq!(m.n_branches_by_order, vec![1, 1]);
        assert_eq!(m.n_tips, 2);
        assert!((m.measured_volume_fraction - 1.0).abs() < 1e-12);
        assert_eq!(m.taper_radii.len(), 10);
    }
}
