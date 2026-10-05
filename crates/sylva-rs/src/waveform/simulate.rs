// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Synthetic waveforms with known echoes.
//!
//! The received waveform of a pulse is the system pulse (a Gaussian of
//! standard deviation `pulse_width` ns) convolved with the targets along
//! the beam (Wagner et al. 2006). A target at range `R` with
//! amplitude `a` and depth `e` (the standard deviation of its extent along
//! the beam, m; 0 for a point target) is a Gaussian in time, so its echo is
//! a Gaussian centred on `2 R / c` of standard deviation
//! `s = sqrt(pulse_width^2 + (e / metres_per_ns)^2)` and, the energy being
//! kept, of peak `a pulse_width / s`. The echoes are summed, added to a
//! constant background with Gaussian noise, and optionally digitised
//! (rounded and clipped to the range of an unsigned integer of `bits`).

#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::{Echoes, Waveforms, C_HALF};
use crate::error::{Error, Result};
use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::transform::normalize;
use crate::Shots;

/// Parameters of [`waveforms_from_shots`].
#[derive(Debug, Clone)]
pub struct SimulateOptions {
    /// Standard deviation of the system pulse (ns).
    pub pulse_width: f64,
    /// Sampling interval (ns).
    pub interval: f64,
    /// Samples per waveform.
    pub n_samples: usize,
    /// Range (m) before a pulse's first echo at which its record starts.
    pub margin: f64,
    /// Fixed range (m) at which every record starts, instead of `margin`.
    pub start_range: Option<f64>,
    pub background: f64,
    /// Standard deviation of the noise (sample units).
    pub noise: f64,
    /// Round and clip the samples to `0 .. 2^bits - 1`.
    pub digitise: bool,
    pub bits: u8,
    /// Amplitude of echoes without an `amplitude` attribute.
    pub amplitude: f64,
    /// Range per ns of round-trip time (m/ns).
    pub metres_per_ns: f64,
    pub seed: u64,
}

impl Default for SimulateOptions {
    fn default() -> Self {
        SimulateOptions {
            pulse_width: 1.5,
            interval: 1.0,
            n_samples: 120,
            margin: 3.0,
            start_range: None,
            background: 10.0,
            noise: 1.0,
            digitise: true,
            bits: 16,
            amplitude: 100.0,
            metres_per_ns: C_HALF,
            seed: 0,
        }
    }
}

/// Waveforms of `shots`: one returning waveform per shot, and the echoes
/// they contain (the truth for a decomposition). Echo amplitudes come from
/// the echo attribute `amplitude` (peak of a point target), depths from
/// `extent` (m). `gps_time` per shot is optional.
pub fn waveforms_from_shots(shots: &Shots, gps_time: Option<&[f64]>, opts: &SimulateOptions) -> Result<(Waveforms, Echoes)> {
    if !(opts.pulse_width > 0.0) || !(opts.interval > 0.0) || !(opts.metres_per_ns > 0.0) {
        return Err(Error::invalid("pulse_width, interval and metres_per_ns must be positive"));
    }
    if opts.n_samples == 0 {
        return Err(Error::invalid("n_samples must be at least 1"));
    }
    if !(opts.noise >= 0.0) || !opts.background.is_finite() {
        return Err(Error::invalid("noise must be zero or positive and background finite"));
    }
    if opts.digitise && !(1..=32).contains(&opts.bits) {
        return Err(Error::invalid("bits must be between 1 and 32"));
    }
    if let Some(t) = gps_time {
        if t.len() != shots.n_shots() {
            return Err(Error::invalid(format!("gps_time has {} values for {} shots", t.len(), shots.n_shots())));
        }
    }
    let n_echo = shots.n_echoes();
    let amp = shots.echo_attrs.get("amplitude").map(|a| a.to_f64()).unwrap_or_else(|| vec![opts.amplitude; n_echo]);
    let ext = shots.echo_attrs.get("extent").map(|a| a.to_f64()).unwrap_or_else(|| vec![0.0; n_echo]);
    if amp.iter().any(|a| !a.is_finite()) || ext.iter().any(|e| !(e.is_finite() && *e >= 0.0)) {
        return Err(Error::invalid("echo amplitudes must be finite and extents finite and non-negative"));
    }
    let mpns = opts.metres_per_ns;
    let n = opts.n_samples;
    let top = if opts.digitise { ((1u64 << opts.bits) - 1) as f64 } else { f64::INFINITY };
    let mut g = Generator::new(opts.seed);
    let mut wf = Waveforms::default();
    let mut truth = Echoes::default();
    wf.samples.reserve(shots.n_shots() * n);
    for s in 0..shots.n_shots() {
        let a = shots.echo_start[s];
        let k = shots.echo_count[s] as usize;
        let start = opts.start_range.unwrap_or_else(|| if k > 0 { shots.echo_range[a] - opts.margin } else { 0.0 });
        let origin = shots.origin[s];
        let dir = normalize(&shots.direction[s]);
        let row = wf.len();
        wf.pulse.push(s as i64);
        wf.gps_time.push(gps_time.map(|t| t[s]).unwrap_or(s as f64));
        wf.origin.push(origin);
        wf.anchor.push(origin);
        wf.direction.push(dir);
        wf.offset.push(start / mpns);
        wf.interval.push(opts.interval);
        wf.metres_per_ns.push(mpns);
        wf.sample_start.push(wf.samples.len());
        wf.sample_count.push(n as u32);
        // The waveform starts as background and every echo adds a Gaussian to
        // it, which is what a digitiser records: overlapping returns sum.
        let mut y = vec![opts.background; n];
        for e in a..a + k {
            let t0 = (shots.echo_range[e] - start) / mpns; // ns after the first sample
            // A target with depth spreads the return: the outgoing pulse and
            // the target's extent combine in quadrature, and the peak falls by
            // as much as the return widens, so the energy under it is kept.
            let depth = ext[e] / mpns;
            let sigma = (opts.pulse_width * opts.pulse_width + depth * depth).sqrt();
            let peak = amp[e] * opts.pulse_width / sigma;
            // Only the samples within eight standard deviations are touched;
            // beyond that the Gaussian is far below one digitiser count.
            let (lo, hi) = (((t0 - 8.0 * sigma) / opts.interval).floor().max(0.0) as usize, ((t0 + 8.0 * sigma) / opts.interval).ceil().max(0.0) as usize);
            for (i, v) in y.iter_mut().enumerate().take(hi.min(n - 1) + 1).skip(lo) {
                let d = i as f64 * opts.interval - t0;
                *v += peak * (-d * d / (2.0 * sigma * sigma)).exp();
            }
            truth.waveform.push(row);
            truth.time.push(t0);
            truth.amplitude.push(peak);
            truth.width.push(sigma);
            truth.xyz.push(wf.position_at(row, t0));
            truth.range.push(shots.echo_range[e]);
        }
        // Noise and digitisation last, so that the recorded echoes above are
        // the truth a decomposition should recover, not what it will see.
        for v in &mut y {
            if opts.noise > 0.0 {
                *v += g.normal(0.0, opts.noise);
            }
            if opts.digitise {
                *v = v.round().clamp(0.0, top);
            }
        }
        wf.samples.extend(y.iter().map(|&v| v as f32));
    }
    wf.attrs.insert("kind".into(), Attr::U8(vec![2; wf.len()]));
    Ok((wf, truth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::decompose::{decompose, DecomposeOptions};

    fn shots(ranges: &[&[f64]]) -> Shots {
        let mut s = Shots::default();
        for r in ranges {
            s.origin.push([0.0, 0.0, 100.0]);
            s.direction.push([0.0, 0.0, -1.0]);
            s.echo_start.push(s.echo_range.len());
            s.echo_count.push(r.len() as u32);
            s.echo_range.extend_from_slice(r);
        }
        s
    }

    #[test]
    fn echoes_land_where_the_targets_are() {
        let s = shots(&[&[80.0, 90.0], &[95.0], &[]]);
        let (wf, truth) = waveforms_from_shots(&s, None, &SimulateOptions { noise: 0.0, digitise: false, ..Default::default() }).unwrap();
        assert_eq!(wf.len(), 3);
        assert_eq!(truth.len(), 3);
        assert!((truth.xyz[0][2] - 20.0).abs() < 1e-9);
        assert!((truth.xyz[1][2] - 10.0).abs() < 1e-9);
        let y = wf.samples_of(0);
        let k = truth.time[0].round() as usize;
        let d = k as f64 - truth.time[0];
        assert!((y[k] as f64 - (10.0 + 100.0 * (-d * d / (2.0 * 1.5 * 1.5)).exp())).abs() < 1e-3);
        let (e, _) = decompose(&wf, &DecomposeOptions { noise: Some(0.1), ..Default::default() }).unwrap();
        assert_eq!(e.len(), 3);
        for i in 0..3 {
            assert!((e.range[i] - truth.range[i]).abs() < 1e-4, "{} {}", e.range[i], truth.range[i]);
        }
    }

    #[test]
    fn seeded_noise_repeats() {
        let s = shots(&[&[50.0]]);
        let o = SimulateOptions { seed: 7, ..Default::default() };
        let (a, _) = waveforms_from_shots(&s, None, &o).unwrap();
        let (b, _) = waveforms_from_shots(&s, None, &o).unwrap();
        assert_eq!(a, b);
        assert!(waveforms_from_shots(&s, None, &SimulateOptions { pulse_width: 0.0, ..Default::default() }).is_err());
    }
}
