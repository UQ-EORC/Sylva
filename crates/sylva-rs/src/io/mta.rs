// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Multiple-time-around (MTA) resolution for RIEGL scans.
//!
//! A scanner firing at megahertz rates has several pulses in flight at once,
//! so an echo may belong to the most recent pulse or to one fired earlier
//! ("zone" 1, 2, 3 ...). RIEGL varies the pulse spacing with a short code
//! (pulse position modulation). Under the *wrong* zone hypothesis two echoes
//! of the same surface seen from neighbouring pulses land on different ranges,
//! because their pulse-interval sums differ; under the right one they land on
//! the same range. That is what this module exploits, in three passes per
//! echo:
//!
//! 1. **Pairwise votes.** Every neighbouring echo (same line within
//!    `window` shots, plus `line_nb` lines each side) is compared under all
//!    hypotheses at once. If exactly one zone puts the two within `tol` of each
//!    other, the neighbour votes for it; if several do (the pulses share the
//!    same interval history) it is uninformative. A clear winner with at least
//!    `min_votes` and no rival above `max_ratio` of it is accepted.
//! 2. **Anchor votes.** Echoes left open are compared with the *resolved*
//!    ranges of the accepted ones in a wider window; a majority of at least
//!    `min_votes2` decides.
//! 3. **Heat map.** The remainder pick the zone whose range falls in a
//!    populated `bucket` of a range histogram of resolved ranges around the
//!    echo (`window3` shots, `line_nb3` lines), when that bucket holds at least
//!    `min_density3` echoes and `ratio3` times the runner-up. The window is
//!    kept local on purpose: a wide one lets the dense near canopy outvote
//!    real far returns and produces points at the wrong range.
//!
//! Echoes that pass none of the three are *MTA uncertain* (`stage == 0`,
//! `zone` = the most probable one); RiSCAN PRO rejects those by default. The defaults below were tuned against rimta5's own result
//! on a VZ-600i forest scan (see docs/guide/io.md): 97-99% of rimta5's point
//! set depending on how much far canopy the lines see, zone choice 98-99.6%.
//!
//! This module is pure Rust and works on per-line tables; the RXP reader in
//! [`super::rxp_mta`] feeds it.

use rayon::prelude::*;

/// Maximum number of zone hypotheses carried per echo.
pub const MAXZ: usize = 8;
/// Echoes per shot stored in the dense grids.
const J: usize = 8;

/// Tuning parameters of [`stage1`] and [`stage23`].
#[derive(Clone, Debug)]
pub struct MtaParams {
    /// Range tolerance (m) for two echoes to count as the same surface.
    pub tol: f32,
    /// Half window in shots along the line, pass 1.
    pub window: usize,
    /// Neighbouring lines each side, pass 1.
    pub line_nb: usize,
    /// Half window in shots within a neighbouring line (passes 1 and 2).
    pub line_window: usize,
    /// Minimum votes for the winner, pass 1.
    pub min_votes: f32,
    /// Largest rival/winner vote ratio still accepted, pass 1.
    pub max_ratio: f32,
    /// Echoes below this reflectance (dB) cast no votes (noise).
    pub weak_db: f32,
    /// Half window in shots, pass 2.
    pub window2: usize,
    /// Neighbouring lines each side, pass 2.
    pub line_nb2: usize,
    /// Minimum anchor votes for the winner, pass 2 (it must also beat the rival).
    pub min_votes2: f32,
    /// Half window in shots, pass 3 (all lines used).
    pub window3: usize,
    /// Neighbouring lines each side, pass 3.
    pub line_nb3: usize,
    /// Range bucket (m) of the pass-3 heat map.
    pub bucket: f32,
    /// Minimum bucket density for the winner, pass 3.
    pub min_density3: f32,
    /// Winner/rival density ratio required, pass 3.
    pub ratio3: f32,
    /// Range extent (m) of the heat map.
    pub max_range: f32,
    /// Acceptance thresholds of zone k >= 3 are multiplied by
    /// `zone_scale^(k-2)`: wrong far hypotheses coincide more easily, and far
    /// zones are rare, so they must show more evidence.
    pub zone_scale: f32,
}

impl Default for MtaParams {
    fn default() -> Self {
        MtaParams {
            tol: 1.0,
            window: 30,
            line_nb: 1,
            line_window: 3,
            min_votes: 8.0,
            max_ratio: 0.25,
            weak_db: -15.0,
            window2: 60,
            line_nb2: 2,
            min_votes2: 2.0,
            window3: 60,
            line_nb3: 4,
            bucket: 2.0,
            min_density3: 2.0,
            ratio3: 4.0,
            max_range: 420.0,
            zone_scale: 2.0,
        }
    }
}

/// Echoes of one scan line, sorted by shot.
#[derive(Clone, Debug, Default)]
pub struct LineEchoes {
    /// Shot index within the line, 0-based, non-decreasing.
    pub shot: Vec<u32>,
    /// Range under each zone hypothesis (m); NaN where RiVLib gave none.
    pub zr: Vec<[f32; MAXZ]>,
    /// Hypotheses in use (zones 1..=nzone).
    pub nzone: usize,
    /// Reflectance (dB); echoes below `weak_db` give no votes.
    pub reflectance: Vec<f32>,
    /// Number of shots in the line (grid rows).
    pub n_shots: usize,
}

/// Per-zone evidence gathered for one echo by the three passes (kept for
/// diagnostics and for training a learned decision rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evidence {
    /// Pass-1 votes from the echo's own line.
    pub votes_own: [f32; MAXZ],
    /// Pass-1 votes from the neighbouring lines.
    pub votes_nb: [f32; MAXZ],
    /// Pass-2 anchor votes.
    pub anchors: [f32; MAXZ],
    /// Pass-3 heat-map density at the hypothesis range.
    pub density: [f32; MAXZ],
}

impl Default for Evidence {
    fn default() -> Self {
        Evidence { votes_own: [-1.0; MAXZ], votes_nb: [-1.0; MAXZ], anchors: [-1.0; MAXZ], density: [-1.0; MAXZ] }
    }
}

/// Outcome for one echo.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Decision {
    /// Best zone, 1-based (the "most probable" one when uncertain; 0 only
    /// when the echo has no hypothesis at all).
    pub zone: u8,
    /// Pass that accepted the echo (1..3); 0 = MTA uncertain.
    pub stage: u8,
    /// Winner's score in that pass.
    pub score: f32,
    /// Runner-up's score in that pass.
    pub rival: f32,
    /// What each pass saw, per zone (-1 where the zone had no hypothesis or the pass did not run).
    pub evidence: Evidence,
}

/// Dense per-line tables used by the passes.
pub struct Grid {
    n_shots: usize,
    k: usize,
    /// (n_shots + 1) x J x MAXZ hypothesis ranges; the last row is NaN padding.
    hyp: Vec<f32>,
    /// (n_shots + 1) x J resolved ranges of anchors (NaN = none).
    res: Vec<f32>,
    /// (n_shots + 2) x nb cumulative heat map of anchors over shots.
    cum: Vec<f32>,
    nb: usize,
    bucket: f32,
}

impl Grid {
    /// Table of hypotheses for one line (anchors are added with [`Grid::set_anchors`]).
    pub fn new(line: &LineEchoes, p: &MtaParams) -> Grid {
        let n = line.n_shots.max(line.shot.last().map(|&s| s as usize + 1).unwrap_or(0));
        let mut hyp = vec![f32::NAN; (n + 1) * J * MAXZ];
        let mut fill = vec![0usize; n + 1];
        for (i, &s) in line.shot.iter().enumerate() {
            let s = s as usize;
            if s >= n || line.reflectance[i] < p.weak_db {
                continue;
            }
            let j = fill[s];
            if j >= J {
                continue;
            }
            fill[s] += 1;
            let base = (s * J + j) * MAXZ;
            hyp[base..base + MAXZ].copy_from_slice(&line.zr[i]);
        }
        let nb = (p.max_range / p.bucket).ceil() as usize + 2;
        Grid { n_shots: n, k: line.nzone.min(MAXZ), hyp, res: vec![f32::NAN; (n + 1) * J], cum: Vec::new(), nb, bucket: p.bucket }
    }

    /// Record the resolved ranges of the accepted echoes of this line.
    pub fn set_anchors(&mut self, line: &LineEchoes, dec: &[Decision], p: &MtaParams) {
        let n = self.n_shots;
        let mut fill = vec![0usize; n + 1];
        let mut heat = vec![0f32; (n + 1) * self.nb];
        for (i, d) in dec.iter().enumerate() {
            if d.stage == 0 || d.zone == 0 || line.reflectance[i] < p.weak_db {
                continue;
            }
            let s = line.shot[i] as usize;
            if s >= n {
                continue;
            }
            let r = line.zr[i][d.zone as usize - 1];
            if !r.is_finite() {
                continue;
            }
            let j = fill[s];
            if j < J {
                fill[s] += 1;
                self.res[s * J + j] = r;
            }
            let b = ((r / self.bucket) as usize).min(self.nb - 1);
            heat[s * self.nb + b] += 1.0;
        }
        // cumulative over shots: cum[s] = sum of rows < s
        let nb = self.nb;
        let mut cum = vec![0f32; (n + 2) * nb];
        for s in 0..=n {
            let (prev, cur) = cum.split_at_mut((s + 1) * nb);
            let prev = &prev[s * nb..];
            for b in 0..nb {
                cur[b] = prev[b] + heat[s * nb + b];
            }
        }
        self.cum = cum;
    }

    #[inline]
    fn hyp_at(&self, s: usize, j: usize) -> &[f32] {
        let base = (s * J + j) * MAXZ;
        &self.hyp[base..base + MAXZ]
    }

    /// Pass-1 votes for echo `zr` at `shot`: neighbours within `w` shots.
    fn votes(&self, shot: i64, zr: &[f32; MAXZ], w: usize, excl_self: bool, tol: f32, out: &mut [f32; MAXZ]) {
        let k = self.k;
        for d in -(w as i64)..=(w as i64) {
            if excl_self && d == 0 {
                continue;
            }
            let s2 = shot + d;
            if s2 < 0 || s2 >= self.n_shots as i64 {
                continue;
            }
            for j in 0..J {
                let c = self.hyp_at(s2 as usize, j);
                if !c[0].is_finite() && !c[1].is_finite() {
                    // empty slot (an echo has at least a zone-1 or zone-2 range)
                    if c.iter().all(|v| !v.is_finite()) {
                        break;
                    }
                }
                let mut hit = usize::MAX;
                let mut nhit = 0usize;
                for z in 0..k {
                    if (c[z] - zr[z]).abs() < tol {
                        nhit += 1;
                        hit = z;
                    }
                }
                if nhit == 1 {
                    out[hit] += 1.0;
                }
            }
        }
    }

    /// Pass-2 votes: anchors within `w` shots whose resolved range matches a hypothesis.
    fn anchor_votes(&self, shot: i64, zr: &[f32; MAXZ], w: usize, excl_self: bool, tol: f32, out: &mut [f32; MAXZ]) {
        let k = self.k;
        for d in -(w as i64)..=(w as i64) {
            if excl_self && d == 0 {
                continue;
            }
            let s2 = shot + d;
            if s2 < 0 || s2 >= self.n_shots as i64 {
                continue;
            }
            let row = &self.res[s2 as usize * J..(s2 as usize + 1) * J];
            for &a in row {
                if !a.is_finite() {
                    break;
                }
                for z in 0..k {
                    if (a - zr[z]).abs() < tol {
                        out[z] += 1.0;
                    }
                }
            }
        }
    }

    /// Pass-3: add this line's anchor heat map over `shot +- w` into `acc`.
    fn heat(&self, shot: i64, w: usize, acc: &mut [f32]) {
        if self.cum.is_empty() {
            return;
        }
        let n = self.n_shots as i64;
        let lo = (shot - w as i64).clamp(0, n) as usize;
        let hi = (shot + w as i64 + 1).clamp(0, n) as usize;
        let nb = self.nb;
        let (a, b) = (&self.cum[lo * nb..(lo + 1) * nb], &self.cum[hi * nb..(hi + 1) * nb]);
        for i in 0..nb {
            acc[i] += b[i] - a[i];
        }
    }
}

/// Threshold multiplier for zone `zone` (1-based).
#[inline]
fn zone_factor(p: &MtaParams, zone: usize) -> f32 {
    p.zone_scale.powi((zone as i32 - 2).max(0))
}

fn best_two(s: &[f32; MAXZ], k: usize) -> (usize, f32, f32) {
    let mut bi = 0;
    let mut b1 = f32::NEG_INFINITY;
    let mut b2 = f32::NEG_INFINITY;
    for (z, &v) in s.iter().enumerate().take(k) {
        if v > b1 {
            b2 = b1;
            b1 = v;
            bi = z;
        } else if v > b2 {
            b2 = v;
        }
    }
    (bi, b1, b2.max(0.0))
}

/// Pass 1 for one line. `grids` holds the line itself at index `line_nb` with
/// `line_nb` neighbours each side (`None` where the scan has no such line).
pub fn stage1(grids: &[Option<&Grid>], line: &LineEchoes, p: &MtaParams) -> Vec<Decision> {
    let c = p.line_nb;
    let own = grids[c].expect("centre grid");
    let k = line.nzone.min(MAXZ);
    (0..line.shot.len())
        .into_par_iter()
        .map(|i| {
            let zr = &line.zr[i];
            let shot = line.shot[i] as i64;
            let mut so = [0f32; MAXZ];
            let mut sn = [0f32; MAXZ];
            own.votes(shot, zr, p.window, true, p.tol, &mut so);
            for (gi, g) in grids.iter().enumerate() {
                if gi == c {
                    continue;
                }
                if let Some(g) = g {
                    g.votes(shot, zr, p.line_window, false, p.tol, &mut sn);
                }
            }
            let mut s = [0f32; MAXZ];
            for z in 0..MAXZ {
                if z < k && zr[z].is_finite() {
                    s[z] = so[z] + sn[z];
                } else {
                    s[z] = -1.0;
                    so[z] = -1.0;
                    sn[z] = -1.0;
                }
            }
            let evidence = Evidence { votes_own: so, votes_nb: sn, ..Default::default() };
            let (bi, b1, b2) = best_two(&s, k);
            let zone = if b1 < 0.0 { 0 } else { bi as u8 + 1 };
            if b1 >= p.min_votes * zone_factor(p, bi + 1) && b2 <= p.max_ratio * b1 {
                Decision { zone, stage: 1, score: b1, rival: b2, evidence }
            } else {
                Decision { zone, stage: 0, score: b1.max(0.0), rival: b2, evidence }
            }
        })
        .collect()
}

/// Pass 2 for the echoes of `line` still undecided after pass 1. `grids`
/// holds the line at index `line_nb2` with `line_nb2` neighbours each side,
/// with pass-1 anchors set ([`Grid::set_anchors`]).
pub fn stage2(grids: &[Option<&Grid>], line: &LineEchoes, dec: &mut [Decision], p: &MtaParams) {
    let c = p.line_nb2;
    let own = grids[c].expect("centre grid");
    let k = line.nzone.min(MAXZ);
    dec.par_iter_mut().enumerate().for_each(|(i, d)| {
        if d.zone == 0 {
            return;
        }
        let decided = d.stage != 0;
        let zr = &line.zr[i];
        let shot = line.shot[i] as i64;
        let mut s = [0f32; MAXZ];
        own.anchor_votes(shot, zr, p.window2, true, p.tol, &mut s);
        for (gi, g) in grids.iter().enumerate() {
            if gi == c {
                continue;
            }
            if let Some(g) = g {
                g.anchor_votes(shot, zr, p.line_window, false, p.tol, &mut s);
            }
        }
        for (z, v) in s.iter_mut().enumerate() {
            if z >= k || !zr[z].is_finite() {
                *v = -1.0;
            }
        }
        d.evidence.anchors = s;
        if decided {
            return;
        }
        let (bi, b1, b2) = best_two(&s, k);
        if b1 >= p.min_votes2 * zone_factor(p, bi + 1) && b1 > b2 {
            d.zone = bi as u8 + 1;
            d.stage = 2;
            d.score = b1;
            d.rival = b2;
        }
    });
}

/// Pass 3 for the echoes of `line` still undecided after pass 2. `grids`
/// holds the line at index `line_nb3` with `line_nb3` neighbours each side,
/// with anchors from passes 1 and 2 set.
pub fn stage3(grids: &[Option<&Grid>], line: &LineEchoes, dec: &mut [Decision], p: &MtaParams) {
    let c = p.line_nb3;
    let own = grids[c].expect("centre grid");
    let k = line.nzone.min(MAXZ);
    let nb = own.nb;
    dec.par_iter_mut().enumerate().for_each(|(i, d)| {
        if d.zone == 0 {
            return;
        }
        let decided = d.stage != 0;
        let zr = &line.zr[i];
        let shot = line.shot[i] as i64;
        let mut acc = vec![0f32; nb];
        for g in grids.iter().flatten() {
            g.heat(shot, p.window3, &mut acc);
        }
        let mut s3 = [-1f32; MAXZ];
        for z in 0..k {
            let r = zr[z];
            if !r.is_finite() {
                continue;
            }
            let b = ((r / p.bucket) as usize).clamp(1, nb - 2);
            s3[z] = acc[b] + 0.5 * (acc[b - 1] + acc[b + 1]);
        }
        d.evidence.density = s3;
        if decided {
            return;
        }
        let (bi, b1, b2) = best_two(&s3, k);
        let zone = if b1 < 0.0 { d.zone } else { bi as u8 + 1 };
        let evidence = d.evidence;
        if b1 >= p.min_density3 * zone_factor(p, bi + 1) && b1 >= p.ratio3 * b2 {
            *d = Decision { zone, stage: 3, score: b1, rival: b2, evidence };
        } else {
            *d = Decision { zone, stage: 0, score: b1.max(0.0), rival: b2, evidence };
        }
    });
}

/// Passes 2 and 3 in one go, with the pass-1 anchors only (convenience for
/// single-window use; the streaming reader refreshes anchors between them).
pub fn stage23(grids: &[Option<&Grid>], line: &LineEchoes, dec: &mut [Decision], p: &MtaParams) {
    let c = p.line_nb3;
    let lo = c.saturating_sub(p.line_nb2);
    let hi = (c + p.line_nb2 + 1).min(grids.len());
    let mut g2: Vec<Option<&Grid>> = vec![None; 2 * p.line_nb2 + 1];
    for (j, g) in grids[lo..hi].iter().enumerate() {
        g2[j + p.line_nb2 - (c - lo)] = *g;
    }
    stage2(&g2, line, dec, p);
    stage3(grids, line, dec, p);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic scan: pulse intervals follow a 7-step code (as ranges, m);
    /// a wall at 20 m (zone 1) across all shots, a far surface at 90 m (zone 2)
    /// on shots 100..300, and a few isolated noise echoes.
    /// Returns the lines and, per echo, (true zone, is isolated noise).
    fn synthetic(n_lines: usize, n_shots: usize) -> (Vec<LineEchoes>, Vec<Vec<(u8, bool)>>) {
        let code = [58.4f32, 65.1, 71.1, 77.7, 74.7, 68.1, 61.4];
        let mut lines = Vec::new();
        let mut truth = Vec::new();
        for l in 0..n_lines {
            let mut tr: Vec<(u8, bool)> = Vec::new();
            // interval before shot s (sum needed for hypotheses)
            let iv: Vec<f32> = (0..n_shots).map(|s| code[(s + 3 * l) % 7]).collect();
            let mut line = LineEchoes { nzone: 3, n_shots, ..Default::default() };
            let push = |s: usize, true_range: f32, true_zone: usize, refl: f32, line: &mut LineEchoes| {
                // recorded zone-1 range = true range minus the intervals of the earlier pulses
                let mut off = 0f32;
                for i in 1..true_zone {
                    off += iv[s - i];
                }
                let r1 = true_range - off;
                let mut zr = [f32::NAN; MAXZ];
                let mut o = 0f32;
                for z in 0..3 {
                    if z > 0 {
                        if s < z {
                            break;
                        }
                        o += iv[s - z];
                    }
                    zr[z] = r1 + o;
                }
                line.shot.push(s as u32);
                line.zr.push(zr);
                line.reflectance.push(refl);
            };
            for s in 3..n_shots {
                let wall = 20.0 + 0.002 * (s as f32) + 0.01 * l as f32;
                push(s, wall, 1, -5.0, &mut line);
                tr.push((1, false));
                if (100..300).contains(&s) {
                    push(s, 90.0 + 0.001 * s as f32, 2, -8.0, &mut line);
                    tr.push((2, false));
                }
                if s % 97 == 5 {
                    push(s, 33.0 + s as f32, 1, -20.0, &mut line); // isolated
                    tr.push((1, true));
                }
            }
            lines.push(line);
            truth.push(tr);
        }
        (lines, truth)
    }

    #[test]
    fn resolves_two_surfaces_and_flags_isolated_echoes() {
        let p = MtaParams::default();
        let (lines, truth) = synthetic(3, 400);
        let mut grids: Vec<Grid> = lines.iter().map(|l| Grid::new(l, &p)).collect();
        let nl = lines.len();
        fn neighbours(i: usize, grids: &[Grid]) -> Vec<Option<&Grid>> {
            let nl = grids.len();
            vec![if i >= 1 { Some(&grids[i - 1]) } else { None }, Some(&grids[i]), if i + 1 < nl { Some(&grids[i + 1]) } else { None }]
        }
        let decs: Vec<Vec<Decision>> = (0..nl).map(|i| stage1(&neighbours(i, &grids), &lines[i], &p)).collect();
        for i in 0..nl {
            grids[i].set_anchors(&lines[i], &decs[i], &p);
        }
        let centre = 1usize;
        let mut dec = decs[centre].clone();
        let mut g3: Vec<Option<&Grid>> = vec![None; 2 * p.line_nb3 + 1];
        for i in 0..nl {
            g3[p.line_nb3 + i - centre] = Some(&grids[i]);
        }
        stage23(&g3, &lines[centre], &mut dec, &p);
        let (mut ok, mut n_wall, mut n_far, mut iso_unc, mut n_iso) = (0, 0, 0, 0, 0);
        for (i, d) in dec.iter().enumerate() {
            let (zone, isolated) = truth[centre][i];
            if isolated {
                n_iso += 1;
                if d.stage == 0 {
                    iso_unc += 1;
                }
            } else {
                if zone == 1 {
                    n_wall += 1;
                } else {
                    n_far += 1;
                }
                if d.stage != 0 && d.zone == zone {
                    ok += 1;
                }
            }
        }
        assert!(n_wall > 390 && n_far > 190, "synthetic set: {n_wall} {n_far}");
        assert!(ok as f64 >= 0.99 * (n_wall + n_far) as f64, "resolved {ok} of {}", n_wall + n_far);
        assert!(iso_unc >= n_iso - 1, "isolated echoes should be uncertain: {iso_unc}/{n_iso}");
    }

    #[test]
    fn best_two_orders() {
        let mut s = [0f32; MAXZ];
        s[0] = 3.0;
        s[1] = 10.0;
        s[2] = 7.0;
        assert_eq!(best_two(&s, 3), (1, 10.0, 7.0));
        assert_eq!(best_two(&s, 1), (0, 3.0, 0.0));
    }
}
