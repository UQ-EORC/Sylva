// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Buttresses and other irregular stem bases as a closed mesh.
//!
//! A cylinder cannot follow a buttressed or fluted base: a circle fitted to a
//! star-shaped cross-section misses the flanges or spans the gaps between
//! them. Here the base is rebuilt volumetrically instead, so any shape works:
//!
//! 1. The points are cut into thin horizontal slices and each is rasterised.
//! 2. A morphological closing bridges the gaps occlusion leaves in the bark,
//!    and flood-filling from outside gives the solid cross-section, however
//!    irregular. Where the outline stays open (part of the bark unseen), the
//!    seen bark is kept, thickened to the closing radius, plus a circle where
//!    the points are a good arc of a round stem.
//! 3. Slices are built from the top down. A buttress only widens towards the
//!    ground, so each section contains the one above, and the part of a slice
//!    kept is the part that connects to the section above: neighbouring stems,
//!    shrubs and logs stay out, and the core carries down where near the
//!    ground only the outsides of the flanges were seen.
//! 4. The buttress ends where the cross-section becomes convex again
//!    (solidity, area over convex-hull area).
//! 5. The stacked cross-sections are turned into a watertight surface by
//!    surface nets (Gibson 1998); the volume is the sum of slice areas.

use std::collections::VecDeque;

use crate::trees::{convex_hull, fit_circle_ransac, polygon_area, RansacCircleParams};
use crate::Point;

/// Settings of [`buttress_mesh`].
#[derive(Debug, Clone)]
pub struct ButtressParams {
    /// Raster cell (m).
    pub resolution: f64,
    /// Slice thickness (m).
    pub slice: f64,
    /// Radius of the closing that bridges gaps in the outline (m).
    pub close_radius: f64,
    /// Points further than this from the stem centre are ignored (m).
    pub max_radius: f64,
    /// Highest possible buttress top, height above ground (m).
    pub max_height: f64,
    /// Buttress top; found from the solidity when `None`.
    pub top: Option<f64>,
    /// Solidity at or above which a cross-section counts as a round stem.
    pub solidity: f64,
    /// Consecutive round slices that end the buttress.
    pub round_run: usize,
    /// Lowest height the top may be found at (m).
    pub min_top: f64,
    /// Slices with fewer points are rebuilt from their neighbours.
    pub min_points: usize,
    /// How fast a section may widen going down (m out per m down). It stops
    /// litter and ground around the base being closed into the solid.
    /// Zero or not finite lifts the limit.
    pub max_flare: f64,
    /// Taubin smoothing passes over the mesh vertices.
    pub smooth: usize,
}

impl Default for ButtressParams {
    fn default() -> Self {
        ButtressParams {
            resolution: 0.02,
            slice: 0.05,
            close_radius: 0.08,
            max_radius: 4.0,
            max_height: 6.0,
            top: None,
            solidity: 0.9,
            round_run: 4,
            min_top: 0.5,
            min_points: 30,
            max_flare: 1.0,
            smooth: 10,
        }
    }
}

/// A buttressed base as a closed mesh, with its cross-section profile.
#[derive(Debug, Clone, Default)]
pub struct Buttress {
    pub vertices: Vec<Point>,
    pub faces: Vec<[u32; 3]>,
    /// Volume from ground to `top` (m³).
    pub volume: f64,
    /// Buttress top, height above ground (m), and as absolute z.
    pub top: f64,
    pub top_z: f64,
    /// Per slice: bottom height (m), area (m²), solidity, and whether the
    /// outline stayed open (part of the bark unseen).
    pub heights: Vec<f64>,
    pub areas: Vec<f64>,
    pub solidities: Vec<f64>,
    pub open: Vec<bool>,
}

struct Grid {
    nx: usize,
    ny: usize,
    x0: f64,
    y0: f64,
    res: f64,
}

impl Grid {
    fn idx(&self, i: usize, j: usize) -> usize {
        i + self.nx * j
    }
}

fn disk(r: isize) -> Vec<(isize, isize)> {
    let mut out = Vec::new();
    for dj in -r..=r {
        for di in -r..=r {
            if di * di + dj * dj <= r * r {
                out.push((di, dj));
            }
        }
    }
    out
}

fn dilate(g: &Grid, a: &[bool], k: &[(isize, isize)]) -> Vec<bool> {
    let mut out = vec![false; a.len()];
    for j in 0..g.ny {
        for i in 0..g.nx {
            if !a[g.idx(i, j)] {
                continue;
            }
            for &(di, dj) in k {
                let (x, y) = (i as isize + di, j as isize + dj);
                if x >= 0 && y >= 0 && (x as usize) < g.nx && (y as usize) < g.ny {
                    out[g.idx(x as usize, y as usize)] = true;
                }
            }
        }
    }
    out
}

fn erode(g: &Grid, a: &[bool], k: &[(isize, isize)]) -> Vec<bool> {
    let inv: Vec<bool> = a.iter().map(|v| !v).collect();
    dilate(g, &inv, k).iter().map(|v| !v).collect()
}

/// Cells reachable from the grid border without crossing `wall` (4-connected).
fn outside(g: &Grid, wall: &[bool]) -> Vec<bool> {
    let mut out = vec![false; wall.len()];
    let mut q = VecDeque::new();
    for i in 0..g.nx {
        for j in [0, g.ny - 1] {
            q.push_back((i, j));
        }
    }
    for j in 0..g.ny {
        for i in [0, g.nx - 1] {
            q.push_back((i, j));
        }
    }
    while let Some((i, j)) = q.pop_front() {
        let k = g.idx(i, j);
        if out[k] || wall[k] {
            continue;
        }
        out[k] = true;
        if i > 0 {
            q.push_back((i - 1, j));
        }
        if j > 0 {
            q.push_back((i, j - 1));
        }
        if i + 1 < g.nx {
            q.push_back((i + 1, j));
        }
        if j + 1 < g.ny {
            q.push_back((i, j + 1));
        }
    }
    out
}

/// The 8-connected component of `a` holding cell `seed` or, failing that,
/// the one nearest to it.
fn component_at(g: &Grid, a: &[bool], seed: (usize, usize)) -> Vec<bool> {
    let mut best = None;
    let mut best_d = f64::INFINITY;
    for j in 0..g.ny {
        for i in 0..g.nx {
            if a[g.idx(i, j)] {
                let d = (i as f64 - seed.0 as f64).hypot(j as f64 - seed.1 as f64);
                if d < best_d {
                    best_d = d;
                    best = Some((i, j));
                }
            }
        }
    }
    let mut out = vec![false; a.len()];
    let Some(start) = best else { return out };
    let mut q = VecDeque::from([start]);
    while let Some((i, j)) = q.pop_front() {
        let k = g.idx(i, j);
        if out[k] || !a[k] {
            continue;
        }
        out[k] = true;
        for dj in -1isize..=1 {
            for di in -1isize..=1 {
                let (x, y) = (i as isize + di, j as isize + dj);
                if x >= 0 && y >= 0 && (x as usize) < g.nx && (y as usize) < g.ny {
                    q.push_back((x as usize, y as usize));
                }
            }
        }
    }
    out
}

/// The 8-connected components of `a` that overlap `b`.
fn keep_touching(g: &Grid, a: &[bool], b: &[bool]) -> Vec<bool> {
    let mut out = vec![false; a.len()];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for j in 0..g.ny {
        for i in 0..g.nx {
            if a[g.idx(i, j)] && b[g.idx(i, j)] {
                q.push_back((i, j));
            }
        }
    }
    while let Some((i, j)) = q.pop_front() {
        let k = g.idx(i, j);
        if out[k] || !a[k] {
            continue;
        }
        out[k] = true;
        for dj in -1isize..=1 {
            for di in -1isize..=1 {
                let (x, y) = (i as isize + di, j as isize + dj);
                if x >= 0 && y >= 0 && (x as usize) < g.nx && (y as usize) < g.ny {
                    q.push_back((x as usize, y as usize));
                }
            }
        }
    }
    out
}

fn solidity(g: &Grid, a: &[bool]) -> (f64, f64) {
    let cell = g.res * g.res;
    let mut corners = Vec::new();
    let mut area = 0.0;
    for j in 0..g.ny {
        for i in 0..g.nx {
            if a[g.idx(i, j)] {
                // The outline runs through the boundary cells: on average half of each is inside.
                let edge = i == 0 || j == 0 || i + 1 == g.nx || j + 1 == g.ny
                    || !a[g.idx(i - 1, j)] || !a[g.idx(i + 1, j)] || !a[g.idx(i, j - 1)] || !a[g.idx(i, j + 1)];
                area += if edge { 0.5 * cell } else { cell };
                let (x, y) = (g.x0 + i as f64 * g.res, g.y0 + j as f64 * g.res);
                corners.extend([[x, y], [x + g.res, y], [x, y + g.res], [x + g.res, y + g.res]]);
            }
        }
    }
    if area == 0.0 {
        return (0.0, 0.0);
    }
    let hull = polygon_area(&convex_hull(&corners));
    (area, if hull > 0.0 { area / hull } else { 0.0 })
}

/// Surface nets over a stack of 2-D masks (`occ[k]` is slice `k`): a watertight
/// quad mesh (returned as triangles) between filled and empty cells.
fn surface_nets(g: &Grid, occ: &[Vec<bool>], dz: f64, z0: f64) -> (Vec<Point>, Vec<[u32; 3]>) {
    let nz = occ.len();
    let at = |i: isize, j: isize, k: isize| -> bool {
        i >= 0 && j >= 0 && k >= 0 && (i as usize) < g.nx && (j as usize) < g.ny && (k as usize) < nz && occ[k as usize][g.idx(i as usize, j as usize)]
    };
    // Dual cells span voxel centres (i-1..i, j-1..j, k-1..k); index by their max corner.
    let (dx, dy, dzn) = (g.nx + 1, g.ny + 1, nz + 1);
    let mut vid = vec![u32::MAX; dx * dy * dzn];
    let mut verts = Vec::new();
    for k in 0..dzn as isize {
        for j in 0..dy as isize {
            for i in 0..dx as isize {
                let mut n_in = 0;
                for c in 0..8 {
                    if at(i - 1 + (c & 1), j - 1 + ((c >> 1) & 1), k - 1 + ((c >> 2) & 1)) {
                        n_in += 1;
                    }
                }
                if n_in == 0 || n_in == 8 {
                    continue;
                }
                vid[i as usize + dx * (j as usize + dy * k as usize)] = verts.len() as u32;
                verts.push([g.x0 + i as f64 * g.res, g.y0 + j as f64 * g.res, z0 + k as f64 * dz]);
            }
        }
    }
    let v = |i: usize, j: usize, k: usize| vid[i + dx * (j + dy * k)];
    let mut faces = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, d: u32, flip: bool| {
        if flip {
            faces.push([a, c, b]);
            faces.push([a, d, c]);
        } else {
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    };
    // A face for each pair of face-adjacent voxels that differ, including
    // where the solid meets the edge of the grid, so the surface closes.
    for k in 0..=nz as isize {
        for j in 0..=g.ny as isize {
            for i in 0..=g.nx as isize {
                let here = at(i, j, k);
                let (iu, ju, ku) = (i as usize, j as usize, k as usize);
                if at(i - 1, j, k) != here && j < g.ny as isize && k < nz as isize {
                    quad(v(iu, ju, ku), v(iu, ju + 1, ku), v(iu, ju + 1, ku + 1), v(iu, ju, ku + 1), here);
                }
                if at(i, j - 1, k) != here && i < g.nx as isize && k < nz as isize {
                    quad(v(iu, ju, ku), v(iu, ju, ku + 1), v(iu + 1, ju, ku + 1), v(iu + 1, ju, ku), here);
                }
                if at(i, j, k - 1) != here && i < g.nx as isize && j < g.ny as isize {
                    quad(v(iu, ju, ku), v(iu + 1, ju, ku), v(iu + 1, ju + 1, ku), v(iu, ju + 1, ku), here);
                }
            }
        }
    }
    (verts, faces)
}

/// Taubin (1995) lambda / mu smoothing: removes the staircase without shrinking.
fn taubin(verts: &mut [Point], faces: &[[u32; 3]], passes: usize) {
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); verts.len()];
    for f in faces {
        for e in 0..3 {
            let (a, b) = (f[e] as usize, f[(e + 1) % 3]);
            if !nb[a].contains(&b) {
                nb[a].push(b);
                nb[b as usize].push(a as u32);
            }
        }
    }
    for _ in 0..passes {
        for w in [0.5, -0.53] {
            let old = verts.to_vec();
            for (i, n) in nb.iter().enumerate() {
                if n.is_empty() {
                    continue;
                }
                let mut m = [0.0; 3];
                for &j in n {
                    for c in 0..3 {
                        m[c] += old[j as usize][c] / n.len() as f64;
                    }
                }
                for c in 0..3 {
                    verts[i][c] = old[i][c] + w * (m[c] - old[i][c]);
                }
            }
        }
    }
}

/// Rebuild the base of one tree from its points.
///
/// `heights` are heights above ground; `(cx, cy)` is the stem centre; `ground_z`
/// is the terrain elevation there, which sets the mesh's absolute z.
pub fn buttress_mesh(points: &[Point], heights: &[f64], cx: f64, cy: f64, ground_z: f64, p: &ButtressParams) -> Buttress {
    let sel: Vec<usize> = (0..points.len())
        .filter(|&i| heights[i] >= 0.0 && heights[i] < p.max_height && (points[i][0] - cx).hypot(points[i][1] - cy) <= p.max_radius)
        .collect();
    let mut out = Buttress::default();
    if sel.len() < p.min_points {
        return out;
    }
    let pad = p.close_radius + 3.0 * p.resolution;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in &sel {
        x0 = x0.min(points[i][0]);
        y0 = y0.min(points[i][1]);
        x1 = x1.max(points[i][0]);
        y1 = y1.max(points[i][1]);
    }
    let g = Grid {
        nx: ((x1 - x0 + 2.0 * pad) / p.resolution).ceil() as usize + 1,
        ny: ((y1 - y0 + 2.0 * pad) / p.resolution).ceil() as usize + 1,
        x0: x0 - pad,
        y0: y0 - pad,
        res: p.resolution,
    };
    let nz = (p.max_height / p.slice).ceil() as usize;
    if let Err(e) = crate::limits::check_cells(
        (g.nx as u128) * (g.ny as u128) * (nz as u128),
        2,
        &format!("a {} x {} x {} buttress raster at {} m", g.nx, g.ny, nz, p.resolution),
        "a coarser resolution, a smaller max_radius, or a lower max_height",
    ) {
        // Nothing here can carry an error back; refusing loudly still beats
        // taking the machine down. The Python layer checks first, so this is
        // the last line rather than the one anybody should meet.
        panic!("{e}");
    }
    let mut slices: Vec<Vec<[f64; 2]>> = vec![Vec::new(); nz];
    for &i in &sel {
        slices[((heights[i] / p.slice) as usize).min(nz - 1)].push([points[i][0], points[i][1]]);
    }
    let kernel = disk((p.close_radius / p.resolution).round().max(1.0) as isize);
    let seed = (((cx - g.x0) / g.res) as usize, ((cy - g.y0) / g.res) as usize);
    let mut open = vec![false; nz];
    let mut filled: Vec<Vec<bool>> = vec![Vec::new(); nz];
    // Top down: a buttress only widens towards the ground, so each section
    // contains the one above, and the part of a slice that belongs to the stem
    // is the part that connects to the section above. That carries the core
    // down where near the ground only the outsides of the flanges were seen.
    // A section may only widen so fast on the way down.
    let flare_cells = (p.max_flare * p.slice / p.resolution).round();
    let flare = (p.max_flare > 0.0 && flare_cells.is_finite() && flare_cells < g.nx.max(g.ny) as f64)
        .then(|| disk((flare_cells as isize).max(1)));
    let task = crate::progress::start("rebuilding the stem base", nz as u64);
    let mut above: Option<Vec<bool>> = None;
    for k in (0..nz).rev() {
        task.inc(1);
        let reach = above.as_ref().filter(|a| a.iter().any(|&v| v)).map(|a| dilate(&g, a, &kernel));
        let limit = flare
            .as_ref()
            .and_then(|f| above.as_ref().filter(|a| a.iter().any(|&v| v)).map(|a| dilate(&g, a, f)));
        let pick = |cells: &[bool]| -> Vec<bool> {
            let touching = reach.as_ref().map(|r| keep_touching(&g, cells, r));
            match touching {
                Some(t) if t.iter().any(|&v| v) => t,
                _ => component_at(&g, cells, seed),
            }
        };
        let mut mask = if slices[k].len() >= p.min_points {
            let mut hit = vec![false; g.nx * g.ny];
            for q in &slices[k] {
                let (i, j) = (((q[0] - g.x0) / g.res) as usize, ((q[1] - g.y0) / g.res) as usize);
                hit[g.idx(i.min(g.nx - 1), j.min(g.ny - 1))] = true;
            }
            let wall = dilate(&g, &hit, &kernel);
            let inside: Vec<bool> = outside(&g, &wall).iter().map(|o| !o).collect();
            let solid = pick(&erode(&g, &inside, &kernel));
            // Did the closing enclose anything beyond the dilated bark itself?
            let enclosed = solid.iter().zip(&wall).filter(|(s, w)| **s && !**w).count();
            if enclosed > 0 {
                solid
            } else {
                // The outline did not close (part of the bark unseen): keep the
                // seen bark thickened by the closing radius, which makes a flange
                // seen only from outside a flange of plausible thickness, plus a
                // circle where the points are a good arc of a round stem.
                open[k] = true;
                let part = pick(&wall);
                let pts: Vec<[f64; 2]> = slices[k]
                    .iter()
                    .copied()
                    .filter(|q| {
                        let (i, j) = (((q[0] - g.x0) / g.res) as usize, ((q[1] - g.y0) / g.res) as usize);
                        part[g.idx(i.min(g.nx - 1), j.min(g.ny - 1))]
                    })
                    .collect();
                let mut m = part;
                let rp = RansacCircleParams { threshold: 0.02, iterations: 200, min_radius: 0.05, max_radius: p.max_radius, seed: k as u64 };
                if let Ok((ccx, ccy, r, inl)) = fit_circle_ransac(&pts, &rp) {
                    if inl.iter().filter(|&&v| v).count() as f64 >= 0.6 * pts.len() as f64 {
                        for j in 0..g.ny {
                            for i in 0..g.nx {
                                let (x, y) = (g.x0 + (i as f64 + 0.5) * g.res, g.y0 + (j as f64 + 0.5) * g.res);
                                if (x - ccx).hypot(y - ccy) <= r {
                                    m[g.idx(i, j)] = true;
                                }
                            }
                        }
                    }
                }
                m
            }
        } else {
            // Too few points: the section above.
            above.clone().unwrap_or_else(|| vec![false; g.nx * g.ny])
        };
        if let Some(l) = &limit {
            for (m, &v) in mask.iter_mut().zip(l) {
                *m &= v;
            }
        }
        if let Some(a) = &above {
            for (m, &v) in mask.iter_mut().zip(a) {
                *m |= v;
            }
        }
        filled[k] = mask.clone();
        above = Some(mask);
    }
    if filled.iter().all(|m| !m.iter().any(|&v| v)) {
        return out;
    }
    let stats: Vec<(f64, f64)> = filled.iter().map(|m| solidity(&g, m)).collect();
    let top_k = match p.top {
        Some(t) => ((t / p.slice).round() as usize).clamp(1, nz),
        None => {
            let start = (p.min_top / p.slice).ceil() as usize;
            (start..nz.saturating_sub(p.round_run))
                .find(|&k| (k..k + p.round_run).all(|m| stats[m].1 >= p.solidity))
                .unwrap_or(nz)
                .max(1)
        }
    };
    out.top = top_k as f64 * p.slice;
    out.top_z = ground_z + out.top;
    out.heights = (0..top_k).map(|k| k as f64 * p.slice).collect();
    out.areas = stats[..top_k].iter().map(|s| s.0).collect();
    out.solidities = stats[..top_k].iter().map(|s| s.1).collect();
    out.open = open[..top_k].to_vec();
    out.volume = out.areas.iter().sum::<f64>() * p.slice;
    let (mut verts, faces) = surface_nets(&g, &filled[..top_k], p.slice, ground_z - 0.5 * p.slice);
    taubin(&mut verts, &faces, p.smooth);
    out.vertices = verts;
    out.faces = faces;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A five-flanged buttress: radius r(theta) = r0 (1 + a cos^8(2.5 theta)) fading with height,
    /// on a stem of radius r0; sampled on its surface with a gap facing away from a scanner.
    fn star_tree(gap: bool) -> (Vec<Point>, Vec<f64>, f64) {
        let r0 = 0.3;
        let radius = |t: f64, h: f64| r0 * (1.0 + 3.0 * (1.0 - h / 2.0).max(0.0) * (2.5 * t).cos().powi(8));
        let mut pts = Vec::new();
        let mut hs = Vec::new();
        let mut volume = 0.0;
        let n_t = 720;
        for k in 0..100 {
            let h = (k as f64 + 0.5) * 0.04;
            let mut area = 0.0;
            for a in 0..n_t {
                let t = a as f64 / n_t as f64 * std::f64::consts::TAU;
                let r = radius(t, h);
                area += 0.5 * r * r * std::f64::consts::TAU / n_t as f64;
                if gap && (0.4..0.9).contains(&t) {
                    continue;
                }
                pts.push([r * t.cos(), r * t.sin(), h]);
                hs.push(h);
            }
            if h < 2.0 {
                volume += area * 0.04;
            }
        }
        (pts, hs, volume)
    }

    #[test]
    fn rebuilds_a_flanged_base_and_its_volume() {
        for gap in [false, true] {
            let (pts, hs, truth) = star_tree(gap);
            let b = buttress_mesh(&pts, &hs, 0.0, 0.0, 0.0, &ButtressParams { top: Some(2.0), ..Default::default() });
            let err = (b.volume - truth).abs() / truth;
            assert!(err < 0.05, "gap {gap}: volume {:.3} vs {truth:.3}", b.volume);
            assert!(!b.faces.is_empty());
            assert!(b.solidities[0] < 0.7, "flanged base should not be convex");
        }
        // The top is found where the section turns round (the flanges fade out at 2 m).
        let (pts, hs, _) = star_tree(false);
        let b = buttress_mesh(&pts, &hs, 0.0, 0.0, 0.0, &ButtressParams::default());
        assert!((b.top - 2.0).abs() < 0.4, "top {}", b.top);
    }
}
