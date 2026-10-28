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

/// Recover the unrecorded pulses of one scan.
///
/// `time_ns` and `direction` describe every recorded pulse in stream order
/// (echoes already grouped; the direction is that of any echo). `zenith_range`
/// is the sweep's first and last zenith (degrees) from the scan pattern;
/// widened to the extremes recorded; `None` uses the extremes alone, which
/// under-fills the sky end of every line when no pulse of the scan returned
/// from there. `stride` keeps every
/// `stride`-th fired pulse, recorded or not, counted along the sweep.
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

    // Sweep direction: the sign the zenith mostly moves in between consecutive pulses.
    let mut dz: Vec<f64> = (1..n).map(|i| zen[i] - zen[i - 1]).filter(|d| d.is_finite()).collect();
    let sweep = if median(&mut dz) >= 0.0 { 1.0 } else { -1.0 };
    drop(dz);
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
    let (sweep_start, sweep_end) = if sweep > 0.0 { (lo, hi) } else { (hi, lo) };

    // Line boundaries: the zenith jumps back against the sweep, across the
    // dead time while the mirror turns. The dead time is required because a
    // stream can hold stray pulses whose direction is out of sequence (a
    // VZ-400i at 600 kHz: one pulse in a thousand, in bursts, jumping 5-90
    // degrees between pulses an interval apart); each one taken for a line
    // would have its sky end filled with thousands of misses. The threshold
    // is a rough interval (the median of every time step, which the
    // within-line steps dominate) scaled far past the scanner's jitter but
    // well short of any line turn.
    let mut all_dt: Vec<f64> = (1..n).map(|i| t[i] - t[i - 1]).collect();
    let dead_time = (20.0 * median(&mut all_dt)).max(50e-6);
    drop(all_dt);
    let mut starts = vec![0usize];
    for i in 1..n {
        if (zen[i] - zen[i - 1]) * sweep < -10.0 && t[i] - t[i - 1] > dead_time {
            starts.push(i);
        }
    }
    starts.push(n);

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

    let mut slot: usize = 0; // fired-pulse counter along the sweep, for the stride
    let mut report = MissReport { n_lines: starts.len() - 1, pulse_interval: interval, zenith_step: step, zenith_range: (lo, hi), ..Default::default() };
    let push_miss = |zen: f64, az: f64, slot: &mut usize, misses: &mut Vec<Point>| {
        if *slot % stride == 0 {
            misses.push(dir(zen, az));
        }
        *slot += 1;
    };
    // The zenith a line's recorded part begins and ends at: the extreme over
    // its first (last) few pulses rather than the first (last) pulse alone,
    // so a stray out-of-sequence pulse at a line's edge cannot set the fill.
    let earliest = |s: &[f64]| s.iter().cloned().fold(if sweep > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }, |m, z| if sweep > 0.0 { m.min(z) } else { m.max(z) });
    let latest = |s: &[f64]| s.iter().cloned().fold(if sweep > 0.0 { f64::NEG_INFINITY } else { f64::INFINITY }, |m, z| if sweep > 0.0 { m.max(z) } else { m.min(z) });
    for w in starts.windows(2) {
        let (a, b) = (w[0], w[1]);
        // Before the first recorded pulse: back along the sweep to its start.
        let z_first = earliest(&zen[a..(a + 8).min(b)]);
        let n_before = (((z_first - sweep_start) * sweep + half) / step.abs()).floor().max(0.0) as usize;
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
        let n_after = (((sweep_end - z_last) * sweep + half) / step.abs()).floor().max(0.0) as usize;
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
        assert_eq!((r.report.n_gap, r.report.n_before, r.report.n_after), (2, 20, 1));
        let got: Vec<(f64, f64)> = r.misses.iter().map(zen_az).collect();
        // the gap: zeniths 30.5 and 30.6 on line 0 (azimuth 10)
        assert!((got[0].0 - 30.5).abs() < 1e-6 && (got[0].1 - 10.0).abs() < 1e-6);
        assert!((got[1].0 - 30.6).abs() < 1e-6);
        // the sky end of line 1: 30.0 .. 31.9 in order, azimuth 11
        assert!((got[2].0 - 30.0).abs() < 1e-6 && (got[2].1 - 11.0).abs() < 1e-6);
        assert!((got[21].0 - 31.9).abs() < 1e-6);
        // the last slot of line 2: 130.0
        assert!((got[22].0 - 130.0).abs() < 1e-6 && (got[22].1 - 12.0).abs() < 1e-6);
        // dead time between lines was not filled
        assert_eq!(r.misses.len(), 23);
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
