// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The pulses a scanner fired but did not record, recovered from the
//! timing and geometry of the pulses it did record.
//!
//! RiVLib's point stream holds only pulses that returned an echo. A
//! RIEGL scanner fires at a fixed repetition rate while its mirror sweeps
//! the zenith at constant speed, one line per azimuth step, so within a
//! line the zenith of a pulse is a linear function of time. That gives
//! every missing pulse a place:
//!
//! - between two recorded pulses of a line, a time gap of `n` intervals
//!   holds `n - 1` misses, at the angles interpolated between them;
//! - before the first recorded pulse of a line (the sky end of the sweep)
//!   and after the last, the sweep continues at the measured angular step
//!   to the pattern's zenith limits;
//! - the dead time between lines, while the mirror turns outside the
//!   field of view, holds nothing.
//!
//! The pulse interval is measured from the data (the adjacent-pulse
//! cluster of consecutive time differences; the scanner's timing jitter
//! cycles over a few pulses and averages out), as is the angular step, so
//! the nominal scan pattern is used only for the zenith limits.
//!
//! This replaces drawing misses at random azimuths along each zenith line
//! ([`Shots::fill_missing`]), which put the right number of misses in the
//! wrong places: a miss drawn into foliage that in fact returned every
//! pulse traces free space through it, and the error grows with density.

use crate::Point;

/// What the reconstruction measured and produced.
#[derive(Debug, Clone, Default)]
pub struct MissReport {
    /// Mirror lines found.
    pub n_lines: usize,
    /// Pulse interval (s).
    pub pulse_interval: f64,
    /// Zenith step between consecutive pulses of a line (degrees, signed by sweep direction).
    pub zenith_step: f64,
    /// Zenith limits of the sweep used (degrees).
    pub zenith_range: (f64, f64),
    /// Misses between recorded pulses of a line.
    pub n_gap: usize,
    /// Misses extrapolated before the first recorded pulse of a line.
    pub n_before: usize,
    /// Misses extrapolated after the last recorded pulse of a line.
    pub n_after: usize,
}

impl MissReport {
    /// Total misses recovered (count): gaps within lines plus those extrapolated before and after.
    pub fn n_misses(&self) -> usize {
        self.n_gap + self.n_before + self.n_after
    }
}

/// Result of [`reconstruct_misses`].
#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    /// Unit directions of the recovered misses, in the frame of the input.
    pub misses: Vec<Point>,
    /// Which recorded pulses fall on a kept stride slot (all true for stride 1).
    pub keep: Vec<bool>,
    /// What the reconstruction measured and how many misses of each kind it produced.
    pub report: MissReport,
}

fn zen_az(d: &Point) -> (f64, f64) {
    let z = (d[2] / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()).clamp(-1.0, 1.0).acos().to_degrees();
    let a = d[0].atan2(d[1]).to_degrees().rem_euclid(360.0);
    (z, a)
}

fn dir(zen: f64, az: f64) -> Point {
    let (z, a) = (zen.to_radians(), az.to_radians());
    [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = v.len() / 2;
    let (_, x, _) = v.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    *x
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    let (_, x, _) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *x
}

/// The scan lines of a stream of pulses in acquisition order.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanLines {
    /// Index of the first pulse of each line, then the number of pulses:
    /// line `i` is `starts[i]..starts[i + 1]`.
    pub starts: Vec<usize>,
    /// `1.0` when the zenith grows along a line (the mirror sweeps from the
    /// sky towards the ground), `-1.0` when it falls.
    pub sweep: f64,
}

impl ScanLines {
    /// Number of scan lines found (count).
    pub fn n_lines(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }
}

/// Split pulses in acquisition order into scan lines.
///
/// A scanner sweeps the zenith along each line and turns its mirror between
/// lines, so a line ends where the zenith (degrees) jumps back against the
/// sweep by more than ten degrees. The sweep direction is the sign the
/// zenith mostly moves in between consecutive pulses. With `time` (seconds
/// per pulse) the jump must also span the dead time of a mirror turn: a
/// stream can hold stray pulses whose direction is out of sequence (a
/// VZ-400i at 600 kHz: one pulse in a thousand, in bursts, jumping 5-90
/// degrees between pulses an interval apart), and each one taken for a line
/// would have its sky end filled with thousands of misses. The dead-time
/// threshold is a rough pulse interval (the median of every time step, which
/// the within-line steps dominate) scaled far past the scanner's jitter but
/// well short of any line turn.
pub fn scan_lines(zenith_deg: &[f64], time: Option<&[f64]>) -> ScanLines {
    let n = zenith_deg.len();
    let mut dz: Vec<f64> = (1..n).map(|i| zenith_deg[i] - zenith_deg[i - 1]).filter(|d| d.is_finite()).collect();
    let sweep = if dz.is_empty() || median(&mut dz) >= 0.0 { 1.0 } else { -1.0 };
    drop(dz);
    let dead_time = time.filter(|_| n > 1).map(|t| {
        let mut all_dt: Vec<f64> = (1..n).map(|i| t[i] - t[i - 1]).collect();
        (20.0 * median(&mut all_dt)).max(50e-6)
    });
    let mut starts = vec![0usize];
    for i in 1..n {
        let back = (zenith_deg[i] - zenith_deg[i - 1]) * sweep < -10.0;
        let turned = match (time, dead_time) {
            (Some(t), Some(d)) => t[i] - t[i - 1] > d,
            _ => true,
        };
        if back && turned {
            starts.push(i);
        }
    }
    starts.push(n);
    ScanLines { starts, sweep }
}

/// Recover the unrecorded pulses of one scan.
///
/// `time_ns` and `direction` describe every recorded pulse in stream order
/// (echoes already grouped; the direction is that of any echo). `zenith_range`
/// is the sweep's first and last zenith (degrees) from the scan pattern;
/// widened to the extremes recorded, and drawn in to where lines typically
/// begin and end when that is within half a degree, since a scanner records
/// its first pulses a few steps after the nominal start; `None` uses the
/// extremes alone, which under-fills the sky end of every line when no pulse
/// of the scan returned from there. A run of at most half a degree missing
/// at either end of a line is the scanner settling, not sky, and is left
/// alone. `stride` keeps every `stride`-th fired pulse, recorded or not,
/// counted along the sweep.
///
/// Returns nothing to add (and `keep` all true) for fewer than two pulses.
pub fn reconstruct_misses(time_ns: &[u64], direction: &[Point], zenith_range: Option<(f64, f64)>, stride: usize) -> Reconstruction {
    let n = time_ns.len();
    let stride = stride.max(1);
    let mut out = Reconstruction { misses: Vec::new(), keep: vec![true; n], report: MissReport::default() };
    if n < 2 {
        return out;
    }
    let (zen, az): (Vec<f64>, Vec<f64>) = direction.iter().map(zen_az).unzip();
    let t: Vec<f64> = time_ns.iter().map(|&x| x as f64 * 1e-9).collect();

    let lines = scan_lines(&zen, Some(&t));
    let (sweep, starts) = (lines.sweep, lines.starts);
    // The sweep's limits: the pattern's, widened to whatever was recorded
    // beyond them (the mirror overshoots the nominal lines by a few steps).
    // The recorded extremes are taken as a 1-in-10,000 quantile and the
    // widening capped at two degrees, so a stray pulse with an out-of-sequence
    // direction cannot push a limit out and have every line filled to it.
    let mut zq: Vec<f64> = zen.iter().cloned().filter(|z| z.is_finite()).collect();
    let (mn, mx) = (quantile(&mut zq, 0.0001), quantile(&mut zq, 0.9999));
    drop(zq);
    let (lo, hi) = match zenith_range {
        Some((a, b)) => ((a.min(b) - 2.0).max(mn).min(a.min(b)), (a.max(b) + 2.0).min(mx).max(a.max(b))),
        None => (mn, mx),
    };
    let (mut sweep_start, mut sweep_end) = if sweep > 0.0 { (lo, hi) } else { (hi, lo) };

    // Pulse interval: the adjacent-pulse cluster of within-line time steps,
    // then refined with every within-line step counted in whole intervals.
    let mut dt: Vec<f64> = Vec::with_capacity(n);
    for w in starts.windows(2) {
        for i in w[0] + 1..w[1] {
            dt.push(t[i] - t[i - 1]);
        }
    }
    if dt.is_empty() {
        return out;
    }
    let q05 = quantile(&mut dt.clone(), 0.05);
    let cluster: Vec<f64> = dt.iter().cloned().filter(|&d| d > 0.0 && d < 1.5 * q05).collect();
    let mut interval = cluster.iter().sum::<f64>() / cluster.len().max(1) as f64;
    if !(interval > 0.0) {
        return out;
    }
    for _ in 0..2 {
        let (mut num, mut den) = (0.0, 0.0);
        for &d in &dt {
            let k = (d / interval).round();
            if (1.0..=50.0).contains(&k) {
                num += d;
                den += k;
            }
        }
        if den > 0.0 {
            interval = num / den;
        }
    }
    // Angular step per pulse: the within-line zenith change per interval.
    let mut steps: Vec<f64> = Vec::with_capacity(n);
    for w in starts.windows(2) {
        for i in w[0] + 1..w[1] {
            let k = ((t[i] - t[i - 1]) / interval).round();
            if k >= 1.0 && (zen[i] - zen[i - 1]).is_finite() {
                steps.push((zen[i] - zen[i - 1]) / k);
            }
        }
    }
    let step = median(&mut steps);
    if !(step.abs() > 0.0) {
        return out;
    }
    let half = step.abs() / 2.0;
    // A within-line time gap longer than a whole sweep is a line boundary the
    // zenith test missed (a line with no recorded pulse between); never fill it.
    let max_gap = ((hi - lo) / step.abs() * 1.05).ceil();

    // The zenith a line's recorded part begins and ends at: the extreme over
    // its first (last) few pulses rather than the first (last) pulse alone,
    // so a stray out-of-sequence pulse at a line's edge cannot set the fill.
    let earliest = |s: &[f64]| s.iter().cloned().fold(if sweep > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }, |m, z| if sweep > 0.0 { m.min(z) } else { m.max(z) });
    let latest = |s: &[f64]| s.iter().cloned().fold(if sweep > 0.0 { f64::NEG_INFINITY } else { f64::INFINITY }, |m, z| if sweep > 0.0 { m.max(z) } else { m.min(z) });
    // Where lines typically begin and end. A scanner records its first pulse
    // a few steps after the pattern's nominal start (a VZ-2000i: 0.15
    // degrees, four pulses), and filling back to the nominal start would
    // draw a ring of misses round every scan. So when most lines begin
    // within half a degree of the limit, the typical beginning is the
    // sweep's start; a larger offset is open sky at the top of every line,
    // which the limit from the scan pattern is there to fill. The same at
    // the end.
    let mut firsts: Vec<f64> = starts.windows(2).map(|w| earliest(&zen[w[0]..(w[0] + 8).min(w[1])])).collect();
    let mut lasts: Vec<f64> = starts.windows(2).map(|w| latest(&zen[w[1].saturating_sub(8).max(w[0])..w[1]])).collect();
    let typical_start = quantile(&mut firsts, if sweep > 0.0 { 0.1 } else { 0.9 });
    let typical_end = quantile(&mut lasts, if sweep > 0.0 { 0.9 } else { 0.1 });
    const SETTLE: f64 = 0.5;
    if (typical_start - sweep_start) * sweep > 0.0 && (typical_start - sweep_start) * sweep <= SETTLE {
        sweep_start = typical_start;
    }
    if (sweep_end - typical_end) * sweep > 0.0 && (sweep_end - typical_end) * sweep <= SETTLE {
        sweep_end = typical_end;
    }
    // Likewise a single line that begins or ends within half a degree of
    // the sweep's limit has merely settled a few steps late or stopped a
    // few early, which drawn as misses would ring every scan with sky; a
    // longer run at a line's end is open sky and is filled.
    let settle_steps = (SETTLE / step.abs()).floor() as usize;
    let (lo, hi) = (sweep_start.min(sweep_end), sweep_start.max(sweep_end));

    let mut slot: usize = 0; // fired-pulse counter along the sweep, for the stride
    let mut report = MissReport { n_lines: starts.len() - 1, pulse_interval: interval, zenith_step: step, zenith_range: (lo, hi), ..Default::default() };
    debug_assert!(lo <= hi);
    let push_miss = |zen: f64, az: f64, slot: &mut usize, misses: &mut Vec<Point>| {
        if *slot % stride == 0 {
            misses.push(dir(zen, az));
        }
        *slot += 1;
    };
    for w in starts.windows(2) {
        let (a, b) = (w[0], w[1]);
        // Before the first recorded pulse: back along the sweep to its start.
        let z_first = earliest(&zen[a..(a + 8).min(b)]);
        let mut n_before = (((z_first - sweep_start) * sweep + half) / step.abs()).floor().max(0.0) as usize;
        if n_before <= settle_steps {
            n_before = 0;
        }
        for k in (1..=n_before).rev() {
            push_miss(z_first - k as f64 * step, az[a], &mut slot, &mut out.misses);
        }
        report.n_before += n_before;
        for i in a..b {
            out.keep[i] = slot % stride == 0;
            slot += 1;
            if i + 1 < b {
                let k = ((t[i + 1] - t[i]) / interval).round();
                if k >= 2.0 && k <= max_gap {
                    let m = k as usize - 1;
                    let daz = {
                        let d = az[i + 1] - az[i];
                        d - 360.0 * (d / 360.0).round()
                    };
                    for j in 1..=m {
                        let f = j as f64 / (m + 1) as f64;
                        push_miss(zen[i] + f * (zen[i + 1] - zen[i]), az[i] + f * daz, &mut slot, &mut out.misses);
                    }
                    report.n_gap += m;
                }
            }
        }
        // After the last recorded pulse: on along the sweep to its end.
        let l = b - 1;
        let z_last = latest(&zen[b.saturating_sub(8).max(a)..b]);
        let mut n_after = (((sweep_end - z_last) * sweep + half) / step.abs()).floor().max(0.0) as usize;
        if n_after <= settle_steps {
            n_after = 0;
        }
        for k in 1..=n_after {
            push_miss(z_last + k as f64 * step, az[l], &mut slot, &mut out.misses);
        }
        report.n_after += n_after;
    }
    out.report = report;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scanner sweeping 30-130 deg in 0.1 deg steps at 1 us per pulse, three
    /// lines 2 ms apart, with `missing` giving the slots (line, index) left out.
    fn scan(missing: &[(usize, usize)]) -> (Vec<u64>, Vec<Point>, usize) {
        let (mut t, mut d) = (Vec::new(), Vec::new());
        let per_line = 1001;
        for line in 0..3 {
            for k in 0..per_line {
                if missing.contains(&(line, k)) {
                    continue;
                }
                t.push((line as u64) * 2_000_000 + (k as u64) * 1000);
                d.push(dir(30.0 + 0.1 * k as f64, 10.0 + line as f64));
            }
        }
        (t, d, per_line)
    }

    #[test]
    fn complete_scan_has_no_misses() {
        let (t, d, _) = scan(&[]);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        assert_eq!(r.report.n_lines, 3);
        assert!((r.report.pulse_interval - 1e-6).abs() < 1e-12);
        assert!((r.report.zenith_step - 0.1).abs() < 1e-9);
        assert_eq!(r.misses.len(), 0);
        assert!(r.keep.iter().all(|&k| k));
    }

    #[test]
    fn gaps_and_line_ends_are_filled_in_place() {
        // line 0: slots 5, 6 missing (a gap); line 1: the first 20 slots (sky); line 2: the last slot.
        let mut missing: Vec<(usize, usize)> = vec![(0, 5), (0, 6), (2, 1000)];
        missing.extend((0..20).map(|k| (1, k)));
        let (t, d, _) = scan(&missing);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        // The single missing slot at the end of line 2 is within the settling allowance.
        assert_eq!((r.report.n_gap, r.report.n_before, r.report.n_after), (2, 20, 0));
        let got: Vec<(f64, f64)> = r.misses.iter().map(zen_az).collect();
        // the gap: zeniths 30.5 and 30.6 on line 0 (azimuth 10)
        assert!((got[0].0 - 30.5).abs() < 1e-6 && (got[0].1 - 10.0).abs() < 1e-6);
        assert!((got[1].0 - 30.6).abs() < 1e-6);
        // the sky end of line 1: 30.0 .. 31.9 in order, azimuth 11
        assert!((got[2].0 - 30.0).abs() < 1e-6 && (got[2].1 - 11.0).abs() < 1e-6);
        assert!((got[21].0 - 31.9).abs() < 1e-6);
        // dead time between lines was not filled
        assert_eq!(r.misses.len(), 22);
    }

    #[test]
    fn stride_counts_fired_pulses() {
        let missing: Vec<(usize, usize)> = (0..20).map(|k| (1, k)).collect();
        let (t, d, per_line) = scan(&missing);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 4);
        // every 4th of the 3003 fired slots, counted on along the sweep
        let kept_recorded = r.keep.iter().filter(|&&k| k).count();
        assert_eq!(kept_recorded + r.misses.len(), (3 * per_line).div_ceil(4));
        assert_eq!(r.misses.len(), 5); // slots 0, 4, 8, 12, 16 of line 1
    }

    #[test]
    fn a_late_but_consistent_line_start_is_not_filled_back_to_the_pattern() {
        // Every line records from slot 3 (30.3 deg) on: the scanner's habit,
        // not sky, so nothing is filled before it; but a line that starts at
        // slot 10 is filled back to the typical start, 30.3.
        let mut missing: Vec<(usize, usize)> = (0..3).flat_map(|l| (0..3).map(move |k| (l, k))).collect();
        missing.extend((3..10).map(|k| (1, k)));
        let (t, d, _) = scan(&missing);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        assert_eq!((r.report.n_before, r.report.n_gap, r.report.n_after), (7, 0, 0));
        assert!((r.report.zenith_range.0 - 30.3).abs() < 1e-9);
        // A line starting four steps late (0.4 deg) has only settled late.
        let (t, d, _) = scan(&(0..4).map(|k| (1, k)).collect::<Vec<_>>());
        assert_eq!(reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1).report.n_before, 0);
        // Starting two degrees late on every line is open sky, filled to the pattern.
        let sky: Vec<(usize, usize)> = (0..3).flat_map(|l| (0..20).map(move |k| (l, k))).collect();
        let (t, d, _) = scan(&sky);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        assert_eq!(r.report.n_before, 60);
    }

    #[test]
    fn a_long_run_of_sky_inside_a_line_is_filled() {
        // Line 1 returns nothing from slot 630 (93 deg) to 997 (129.7 deg): a
        // tilted scan looking up into open sky, its last three pulses on the
        // scanner's own base. All 368 missing pulses are gaps, not line ends.
        let missing: Vec<(usize, usize)> = (630..998).map(|k| (1, k)).collect();
        let (t, d, _) = scan(&missing);
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        assert_eq!((r.report.n_gap, r.report.n_before, r.report.n_after), (368, 0, 0));
        let zen: Vec<f64> = r.misses.iter().map(|m| zen_az(m).0).collect();
        assert!((zen[0] - 93.0).abs() < 1e-6 && (zen[367] - 129.7).abs() < 1e-6);
    }

    #[test]
    fn scan_lines_split_on_the_zenith_turn_with_or_without_time() {
        // Three lines sweeping 30-150 degrees, the third cut short.
        let zen: Vec<f64> = (0..5).map(|i| 30.0 + 30.0 * i as f64).cycle().take(13).collect();
        let l = scan_lines(&zen, None);
        assert_eq!((l.n_lines(), l.sweep, l.starts), (3, 1.0, vec![0, 5, 10, 13]));
        // The same sweep downwards.
        let down: Vec<f64> = zen.iter().map(|z| 180.0 - z).collect();
        assert_eq!(scan_lines(&down, None).starts, vec![0, 5, 10, 13]);
        // With time, a jump back within the dead time is a stray pulse, not a line.
        let mut t: Vec<f64> = (0..13).map(|i| i as f64 * 1e-5).collect();
        t[5] += 1e-3;
        assert_eq!(scan_lines(&zen, Some(&t)).starts, vec![0, 5, 13]);
        assert_eq!(scan_lines(&[], None).starts, vec![0, 0]);
        assert_eq!(scan_lines(&[45.0], Some(&[0.0])).starts, vec![0, 1]);
    }

    #[test]
    fn stray_out_of_sequence_pulses_do_not_start_lines() {
        // A VZ-400i stream can carry pulses whose direction is out of sequence:
        // mid-line, with the normal pulse interval, the zenith jumps back by tens
        // of degrees for a pulse or two. They are not line turns (no dead time)
        // and must not have a sky end filled; nor may one at a line's edge set it.
        let (t, mut d, _) = scan(&[]);
        d[500] = dir(45.0, 10.0); // line 0, slot 500: nominal zenith 80, recorded 45
        d[501] = dir(120.0, 10.0);
        d[1001] = dir(95.0, 11.0); // the first pulse of line 1 (nominal 30)
        let r = reconstruct_misses(&t, &d, Some((30.0, 130.0)), 1);
        assert_eq!(r.report.n_lines, 3);
        assert_eq!((r.report.n_gap, r.report.n_after), (0, 0));
        assert!(r.report.n_before <= 1); // the stray first pulse of line 1 can cost one miss, never a sky end
        assert!((r.report.zenith_step - 0.1).abs() < 1e-6);
    }

    #[test]
    fn limits_inferred_from_data_when_no_pattern() {
        let missing: Vec<(usize, usize)> = (0..20).map(|k| (1, k)).collect();
        let (t, d, _) = scan(&missing);
        let r = reconstruct_misses(&t, &d, None, 1);
        assert!((r.report.zenith_range.0 - 30.0).abs() < 1e-9 && (r.report.zenith_range.1 - 130.0).abs() < 1e-9);
        assert_eq!(r.report.n_before, 20);
    }
}
