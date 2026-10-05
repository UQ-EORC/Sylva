// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Gaussian decomposition of waveforms (Hofton et al. 2000; Wagner et al.
//! 2006).
//!
//! Each waveform is modelled as a constant background plus a sum of
//! Gaussians, one per echo. The background and the noise are estimated
//! robustly from the samples (median and median absolute deviation, with
//! the samples of echoes clipped away); the waveform is smoothed with a
//! Gaussian kernel; candidate echoes are taken from the smoothed signal,
//! either at the zero crossings of its first derivative (local maxima, as
//! Wagner et al. 2006) or between the inflection points around each local
//! minimum of its second derivative (Hofton et al. 2000), which also finds
//! echoes that overlap too much to make a maximum of their own. Candidates
//! weaker than the detection threshold are dropped, and the amplitudes,
//! positions and widths of the rest are fitted to the raw samples with the
//! Levenberg-Marquardt algorithm (Levenberg 1944; Marquardt 1963). Fitted
//! echoes weaker than the threshold are removed, echoes closer than half
//! the narrowest allowed width are merged, and, when `refine` is on, an
//! echo is added wherever the smoothed residual still exceeds the
//! threshold; the fit is repeated until nothing changes.
//!
//! Waveforms are independent, so they are processed in parallel and the
//! result does not depend on the number of threads.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use super::{Echoes, Waveforms};
use crate::error::{Error, Result};
use crate::transform::{dot, normalize, sub};

/// How candidate echoes are found on the smoothed waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakMethod {
    /// Local maxima: zero crossings of the first derivative.
    Derivative,
    /// Local minima of the second derivative, between inflection points.
    Inflection,
}

/// Parameters of [`decompose`].
#[derive(Debug, Clone)]
pub struct DecomposeOptions {
    /// Standard deviation of the smoothing kernel (ns); 0 for none.
    pub smooth: f64,
    /// Detection threshold, in standard deviations of the noise.
    pub threshold: f64,
    /// Smallest amplitude of an echo (sample units), whatever the noise.
    pub min_amplitude: f64,
    pub peaks: PeakMethod,
    /// Most echoes per waveform.
    pub max_echoes: usize,
    /// Bounds on the fitted Gaussian standard deviation (ns).
    pub min_width: f64,
    pub max_width: f64,
    /// Noise standard deviation to use instead of the estimate.
    pub noise: Option<f64>,
    /// Background level to use instead of the estimate.
    pub background: Option<f64>,
    /// Levenberg-Marquardt iterations per fit.
    pub max_iter: usize,
    /// Add echoes where the residual of the fit still shows one.
    pub refine: bool,
}

impl Default for DecomposeOptions {
    fn default() -> Self {
        DecomposeOptions {
            smooth: 1.0,
            threshold: 4.0,
            min_amplitude: 0.0,
            peaks: PeakMethod::Inflection,
            max_echoes: 10,
            min_width: 0.3,
            max_width: 20.0,
            noise: None,
            background: None,
            max_iter: 100,
            refine: true,
        }
    }
}

impl DecomposeOptions {
    pub fn validate(&self) -> Result<()> {
        if !(self.smooth >= 0.0) || !self.smooth.is_finite() {
            return Err(Error::invalid(format!("smooth must be zero or positive, got {}", self.smooth)));
        }
        if !(self.threshold >= 0.0) || !(self.min_amplitude >= 0.0) {
            return Err(Error::invalid("threshold and min_amplitude must be zero or positive"));
        }
        if !(self.min_width > 0.0) || !(self.max_width > self.min_width) || !self.max_width.is_finite() {
            return Err(Error::invalid(format!("widths must satisfy 0 < min_width < max_width, got {} and {}", self.min_width, self.max_width)));
        }
        if self.max_echoes == 0 {
            return Err(Error::invalid("max_echoes must be at least 1"));
        }
        if let Some(n) = self.noise {
            if !(n >= 0.0) || !n.is_finite() {
                return Err(Error::invalid(format!("noise must be zero or positive, got {n}")));
            }
        }
        if let Some(b) = self.background {
            if !b.is_finite() {
                return Err(Error::invalid("background must be finite"));
            }
        }
        Ok(())
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        0.5 * (v[m / 2 - 1] + v[m / 2])
    }
}

/// Background level and noise standard deviation of a waveform.
///
/// A robust start (median, and 1.4826 times the median absolute deviation,
/// recomputed without the samples more than 3 standard deviations above
/// the median until they settle, so that echoes are clipped away), then the
/// mean and standard deviation of the samples within 3 standard deviations
/// of the background, the latter divided by 0.9866 for the clipped tails of
/// a normal distribution, iterated. The second stage is what keeps the
/// estimate right for digitised samples, whose median absolute deviation
/// moves in whole counts. NaN samples are ignored; NaN if none is left.
pub fn estimate_noise(samples: &[f32]) -> (f64, f64) {
    let all: Vec<f64> = samples.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();
    if all.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut kept = all.clone();
    let mut bg = f64::NAN;
    let mut sd = f64::NAN;
    for _ in 0..20 {
        let mut work = kept.clone();
        let b = median(&mut work);
        let mut dev: Vec<f64> = kept.iter().map(|v| (v - b).abs()).collect();
        let s = 1.4826 * median(&mut dev);
        let next: Vec<f64> = kept.iter().cloned().filter(|v| v - b <= 3.0 * s.max(f64::MIN_POSITIVE)).collect();
        let settled = next.len() == kept.len();
        bg = b;
        sd = s;
        if settled || next.len() < 3 {
            break;
        }
        kept = next;
    }
    let moments = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = if v.len() > 1 { (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt() } else { 0.0 };
        (m, s)
    };
    if sd == 0.0 {
        let (m, s) = moments(&kept);
        return (if s > 0.0 { m } else { bg }, s);
    }
    for _ in 0..10 {
        let near: Vec<f64> = all.iter().cloned().filter(|v| (v - bg).abs() <= 3.0 * sd).collect();
        if near.len() < 3 {
            break;
        }
        let (m, s) = moments(&near);
        let s = s / 0.9866;
        let done = (m - bg).abs() <= 1e-9 * (1.0 + m.abs()) && (s - sd).abs() <= 1e-9 * (1.0 + s);
        bg = m;
        sd = s;
        if done || sd == 0.0 {
            break;
        }
    }
    (bg, sd)
}

/// Convolve with a Gaussian of standard deviation `sigma` samples,
/// truncated at 4 sigma and renormalised at the ends.
pub fn smooth(y: &[f64], sigma: f64) -> Vec<f64> {
    if !(sigma > 0.0) || y.len() < 2 {
        return y.to_vec();
    }
    let h = (4.0 * sigma).ceil() as isize;
    let k: Vec<f64> = (-h..=h).map(|j| (-(j * j) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let n = y.len() as isize;
    (0..n)
        .map(|i| {
            let (mut s, mut w) = (0.0, 0.0);
            for j in -h..=h {
                let t = i + j;
                if t >= 0 && t < n && y[t as usize].is_finite() {
                    let kw = k[(j + h) as usize];
                    s += kw * y[t as usize];
                    w += kw;
                }
            }
            if w > 0.0 {
                s / w
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// A candidate or fitted echo, in samples: centre, amplitude, standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    pub centre: f64,
    pub amplitude: f64,
    pub sigma: f64,
}

fn parabolic(a: f64, b: f64, c: f64) -> f64 {
    let d = a - 2.0 * b + c;
    if d.abs() > 0.0 {
        (0.5 * (a - c) / d).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Where `f` crosses `level` between samples `i` and `i + 1` (linear).
fn crossing(f: &[f64], i: usize, level: f64) -> f64 {
    let (a, b) = (f[i], f[i + 1]);
    if b != a {
        i as f64 + ((level - a) / (b - a)).clamp(0.0, 1.0)
    } else {
        i as f64 + 0.5
    }
}

/// Candidate echoes on a smoothed, background-free waveform `s`: those
/// whose smoothed amplitude exceeds `thr`. `smooth_sigma` (samples) is taken
/// out of the width estimates.
pub fn find_peaks(s: &[f64], thr: f64, method: PeakMethod, smooth_sigma: f64) -> Vec<Component> {
    let n = s.len();
    let mut out = Vec::new();
    if n < 3 {
        return out;
    }
    let deconv = |w: f64| (w * w - smooth_sigma * smooth_sigma).max(0.0).sqrt();
    match method {
        PeakMethod::Derivative => {
            let mut i = 1;
            while i + 1 < n {
                if s[i] > s[i - 1] && s[i] >= s[i + 1] && s[i] > thr {
                    // A plateau: its middle.
                    let mut j = i;
                    while j + 1 < n && s[j + 1] == s[i] {
                        j += 1;
                    }
                    if j + 1 < n && s[j + 1] > s[i] {
                        i = j + 1;
                        continue;
                    }
                    let c = if j > i { 0.5 * (i + j) as f64 } else { i as f64 + parabolic(s[i - 1], s[i], s[i + 1]) };
                    let half = 0.5 * s[i];
                    let mut l = i;
                    while l > 0 && s[l] > half {
                        l -= 1;
                    }
                    let mut r = j;
                    while r + 1 < n && s[r] > half {
                        r += 1;
                    }
                    let lx = if s[l] <= half { crossing(s, l, half) } else { 0.0 };
                    let rx = if s[r] <= half && r > 0 { crossing(s, r - 1, half) } else { (n - 1) as f64 };
                    let sigma = deconv((rx - lx) / 2.354_820_045);
                    out.push(Component { centre: c, amplitude: s[i], sigma });
                    i = j + 1;
                } else {
                    i += 1;
                }
            }
        }
        PeakMethod::Inflection => {
            let mut dd = vec![0.0; n];
            for i in 1..n - 1 {
                dd[i] = s[i - 1] - 2.0 * s[i] + s[i + 1];
            }
            dd[0] = dd[1];
            dd[n - 1] = dd[n - 2];
            let mut i = 1;
            while i + 1 < n {
                if dd[i] < 0.0 && dd[i] < dd[i - 1] && dd[i] <= dd[i + 1] {
                    let mut j = i;
                    while j + 1 < n && dd[j + 1] == dd[i] {
                        j += 1;
                    }
                    let c = if j > i { 0.5 * (i + j) as f64 } else { i as f64 + parabolic(dd[i - 1], dd[i], dd[i + 1]) };
                    let k = c.round().clamp(0.0, (n - 1) as f64) as usize;
                    let amp = s[k].max(s[i]);
                    if amp > thr {
                        let mut l = i;
                        while l > 0 && dd[l] < 0.0 {
                            l -= 1;
                        }
                        let mut r = j;
                        while r + 1 < n && dd[r] < 0.0 {
                            r += 1;
                        }
                        let lx = if dd[l] >= 0.0 { crossing(&dd, l, 0.0) } else { 0.0 };
                        let rx = if dd[r] >= 0.0 && r > 0 { crossing(&dd, r - 1, 0.0) } else { (n - 1) as f64 };
                        let sigma = deconv((rx - lx) / 2.0);
                        out.push(Component { centre: c, amplitude: amp, sigma });
                    }
                    i = j + 1;
                } else {
                    i += 1;
                }
            }
        }
    }
    out
}

fn model_into(t: &[f64], comps: &[Component], out: &mut [f64]) {
    out.fill(0.0);
    if t.is_empty() {
        return;
    }
    for c in comps {
        let inv = 1.0 / (2.0 * c.sigma * c.sigma);
        let reach = 8.0 * c.sigma;
        let lo = ((c.centre - reach).floor().max(0.0)) as usize;
        let hi = ((c.centre + reach).ceil().max(0.0) as usize).min(t.len().saturating_sub(1));
        for i in lo..=hi.min(t.len().saturating_sub(1)) {
            let d = t[i] - c.centre;
            out[i] += c.amplitude * (-d * d * inv).exp();
        }
    }
}

fn cost(y: &[f64], t: &[f64], comps: &[Component], buf: &mut [f64]) -> f64 {
    model_into(t, comps, buf);
    y.iter().zip(buf.iter()).filter(|(v, _)| v.is_finite()).map(|(v, m)| (v - m) * (v - m)).sum()
}

/// Least-squares fit of Gaussians to `y` (background removed) with
/// Levenberg-Marquardt and an analytic Jacobian. Amplitudes stay positive,
/// widths within `[min_sigma, max_sigma]` and centres within the waveform.
/// Returns the fitted components, the residual sum of squares and the
/// number of iterations.
pub fn fit_gaussians(y: &[f64], init: &[Component], min_sigma: f64, max_sigma: f64, max_iter: usize) -> (Vec<Component>, f64, usize) {
    let n = y.len();
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut comps: Vec<Component> = init
        .iter()
        .map(|c| Component { centre: c.centre.clamp(0.0, (n.max(1) - 1) as f64), amplitude: c.amplitude.max(1e-12), sigma: c.sigma.clamp(min_sigma, max_sigma) })
        .collect();
    let mut buf = vec![0.0; n];
    let mut c0 = cost(y, &t, &comps, &mut buf);
    if comps.is_empty() || n == 0 {
        return (comps, c0, 0);
    }
    let m = 3 * comps.len();
    let mut lambda = 1e-3;
    let mut iters = 0;
    let clamp = |c: &mut Component| {
        c.amplitude = c.amplitude.max(1e-12);
        c.sigma = c.sigma.clamp(min_sigma, max_sigma);
        c.centre = c.centre.clamp(-0.5, n as f64 - 0.5);
    };
    for it in 0..max_iter {
        iters = it + 1;
        // J^T J and J^T r, accumulated row by row.
        let mut jtj = DMatrix::<f64>::zeros(m, m);
        let mut jtr = DVector::<f64>::zeros(m);
        model_into(&t, &comps, &mut buf);
        let mut row = vec![0.0; m];
        for i in 0..n {
            if !y[i].is_finite() {
                continue;
            }
            let r = y[i] - buf[i];
            for (k, c) in comps.iter().enumerate() {
                let d = t[i] - c.centre;
                let z = d * d / (2.0 * c.sigma * c.sigma);
                if z > 40.0 {
                    row[3 * k] = 0.0;
                    row[3 * k + 1] = 0.0;
                    row[3 * k + 2] = 0.0;
                    continue;
                }
                let e = (-z).exp();
                row[3 * k] = e;
                row[3 * k + 1] = c.amplitude * e * d / (c.sigma * c.sigma);
                row[3 * k + 2] = c.amplitude * e * d * d / (c.sigma * c.sigma * c.sigma);
            }
            for a in 0..m {
                if row[a] == 0.0 {
                    continue;
                }
                jtr[a] += row[a] * r;
                for b in a..m {
                    jtj[(a, b)] += row[a] * row[b];
                }
            }
        }
        for a in 0..m {
            for b in 0..a {
                jtj[(a, b)] = jtj[(b, a)];
            }
        }
        if jtr.amax() <= 1e-14 * (1.0 + c0) {
            break;
        }
        let mut improved = false;
        let mut small_step = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for j in 0..m {
                a[(j, j)] += lambda * jtj[(j, j)].max(1e-12);
            }
            let Some(delta) = a.cholesky().map(|ch| ch.solve(&jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let mut trial = comps.clone();
            for (k, c) in trial.iter_mut().enumerate() {
                c.amplitude += delta[3 * k];
                c.centre += delta[3 * k + 1];
                c.sigma += delta[3 * k + 2];
                clamp(c);
            }
            let c1 = cost(y, &t, &trial, &mut buf);
            if c1 < c0 {
                small_step = (c0 - c1) <= 1e-9 * c0.max(f64::MIN_POSITIVE) || delta.amax() < 1e-7;
                comps = trial;
                c0 = c1;
                lambda = (lambda / 10.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved || small_step {
            break;
        }
    }
    (comps, c0, iters)
}

/// Fit of one waveform.
#[derive(Debug, Clone, Default)]
pub struct WaveFit {
    /// Echoes in samples (centre after the first sample), ascending.
    pub components: Vec<Component>,
    pub background: f64,
    pub noise: f64,
    /// Root mean square residual of the fit (sample units).
    pub rmse: f64,
    pub iterations: usize,
}

/// Decompose one waveform (`interval` in ns).
pub fn decompose_one(samples: &[f32], interval: f64, opts: &DecomposeOptions) -> WaveFit {
    let n = samples.len();
    let (bg_est, sd_est) = if opts.background.is_some() && opts.noise.is_some() { (f64::NAN, f64::NAN) } else { estimate_noise(samples) };
    let bg = opts.background.unwrap_or(bg_est);
    let sd = opts.noise.unwrap_or(sd_est);
    let mut fit = WaveFit { background: bg, noise: sd, rmse: f64::NAN, ..Default::default() };
    if n < 3 || !bg.is_finite() {
        return fit;
    }
    let y: Vec<f64> = samples.iter().map(|&v| v as f64 - bg).collect();
    let thr = (opts.threshold * if sd.is_finite() { sd } else { 0.0 }).max(opts.min_amplitude);
    let ss = opts.smooth / interval;
    let s = smooth(&y, ss);
    let (min_s, max_s) = (opts.min_width / interval, opts.max_width / interval);
    let mut comps = find_peaks(&s, thr, opts.peaks, ss);
    // Smoothing lowers a peak by sigma / sqrt(sigma^2 + smooth^2); undo it for the start value.
    for c in &mut comps {
        c.sigma = c.sigma.clamp(min_s, max_s);
        c.amplitude *= (c.sigma * c.sigma + ss * ss).sqrt() / c.sigma;
    }
    if comps.len() > opts.max_echoes {
        comps.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
        comps.truncate(opts.max_echoes);
    }
    comps.sort_by(|a, b| a.centre.total_cmp(&b.centre));
    if comps.is_empty() {
        let ss_res: f64 = y.iter().filter(|v| v.is_finite()).map(|v| v * v).sum();
        fit.rmse = (ss_res / n as f64).sqrt();
        return fit;
    }
    let mut rss;
    let mut total_iter = 0;
    let mut rounds = 0;
    let mut added: usize = 0;
    // A refinement whose echo the next fit drops would be proposed again: stop refining then.
    let mut just_added = false;
    let mut refine = opts.refine;
    loop {
        rounds += 1;
        let (c, r, it) = fit_gaussians(&y, &comps, min_s, max_s, opts.max_iter);
        total_iter += it;
        rss = r;
        comps = c;
        comps.sort_by(|a, b| a.centre.total_cmp(&b.centre));
        // Drop weak and out-of-window echoes; merge coincident ones.
        let before = comps.len();
        comps.retain(|c| c.amplitude >= thr && c.amplitude > 0.0 && c.centre >= -0.5 && c.centre <= n as f64 - 0.5);
        let mut merged: Vec<Component> = Vec::with_capacity(comps.len());
        for c in comps.drain(..) {
            if let Some(last) = merged.last_mut() {
                if (c.centre - last.centre).abs() < 0.5 * min_s.max(0.5) {
                    let w = last.amplitude + c.amplitude;
                    last.centre = (last.centre * last.amplitude + c.centre * c.amplitude) / w;
                    last.sigma = last.sigma.max(c.sigma);
                    last.amplitude = last.amplitude.max(c.amplitude);
                    continue;
                }
            }
            merged.push(c);
        }
        comps = merged;
        let mut changed = comps.len() != before;
        if changed && just_added {
            refine = false;
        }
        just_added = false;
        if !changed && refine && comps.len() < opts.max_echoes && added < opts.max_echoes {
            let mut model = vec![0.0; n];
            let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
            model_into(&t, &comps, &mut model);
            let resid: Vec<f64> = y.iter().zip(&model).map(|(a, b)| a - b).collect();
            let rs = smooth(&resid, ss.max(1.0));
            let cand = find_peaks(&rs, thr, PeakMethod::Derivative, ss.max(1.0));
            if let Some(best) = cand.iter().filter(|c| c.amplitude > thr).max_by(|a, b| a.amplitude.total_cmp(&b.amplitude)) {
                let far = comps.iter().all(|c| (c.centre - best.centre).abs() > 0.5 * c.sigma.max(min_s));
                if far {
                    comps.push(Component { centre: best.centre, amplitude: best.amplitude, sigma: best.sigma.clamp(min_s, max_s) });
                    comps.sort_by(|a, b| a.centre.total_cmp(&b.centre));
                    added += 1;
                    just_added = true;
                    changed = true;
                }
            }
        }
        if !changed || comps.is_empty() || rounds >= 2 * opts.max_echoes + 5 {
            break;
        }
    }
    if comps.is_empty() {
        rss = y.iter().filter(|v| v.is_finite()).map(|v| v * v).sum();
    }
    fit.rmse = (rss / n as f64).sqrt();
    fit.components = comps;
    fit.iterations = total_iter;
    fit
}

/// Per-waveform results of [`decompose`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitStats {
    pub background: Vec<f64>,
    pub noise: Vec<f64>,
    pub rmse: Vec<f64>,
    pub n_echoes: Vec<u32>,
    pub iterations: Vec<u32>,
}

/// Decompose every waveform, in parallel.
pub fn decompose(wf: &Waveforms, opts: &DecomposeOptions) -> Result<(Echoes, FitStats)> {
    opts.validate()?;
    wf.validate()?;
    let fits: Vec<WaveFit> = (0..wf.len()).into_par_iter().with_min_len(64).map(|i| decompose_one(wf.samples_of(i), wf.interval[i], opts)).collect();
    let mut e = Echoes::default();
    let mut st = FitStats::default();
    for (i, f) in fits.iter().enumerate() {
        let dir = normalize(&wf.direction[i]);
        let origin = wf.origin[i];
        let has_origin = origin.iter().all(|v| v.is_finite());
        for c in &f.components {
            let t = c.centre * wf.interval[i];
            let p = wf.position_at(i, t);
            e.waveform.push(i);
            e.time.push(t);
            e.amplitude.push(c.amplitude);
            e.width.push(c.sigma * wf.interval[i]);
            e.xyz.push(p);
            e.range.push(if has_origin { dot(&sub(&p, &origin), &dir) } else { f64::NAN });
        }
        st.background.push(f.background);
        st.noise.push(f.noise);
        st.rmse.push(f.rmse);
        st.n_echoes.push(f.components.len() as u32);
        st.iterations.push(f.iterations as u32);
    }
    Ok((e, st))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(n: usize, comps: &[(f64, f64, f64)], bg: f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                (bg + comps.iter().map(|&(c, a, s)| a * (-(t - c) * (t - c) / (2.0 * s * s)).exp()).sum::<f64>()) as f32
            })
            .collect()
    }

    #[test]
    fn noise_estimate_ignores_echoes() {
        let mut g = crate::util::nprandom::Generator::new(3);
        let mut y = gauss(200, &[(50.0, 200.0, 2.0), (120.0, 80.0, 3.0)], 10.0);
        for v in &mut y {
            *v += g.normal(0.0, 2.0) as f32;
        }
        let (b, s) = estimate_noise(&y);
        assert!((b - 10.0).abs() < 0.6, "{b}");
        assert!((s - 2.0).abs() < 0.4, "{s}");
    }

    #[test]
    fn noiseless_echoes_are_recovered() {
        let truth = [(30.0, 100.0, 2.0), (45.5, 60.0, 2.5), (80.25, 20.0, 1.8)];
        let y = gauss(120, &truth, 5.0);
        for method in [PeakMethod::Derivative, PeakMethod::Inflection] {
            let opts = DecomposeOptions { peaks: method, noise: Some(0.5), ..Default::default() };
            let f = decompose_one(&y, 1.0, &opts);
            assert_eq!(f.components.len(), 3, "{method:?} {:?}", f.components);
            for (c, t) in f.components.iter().zip(&truth) {
                assert!((c.centre - t.0).abs() < 1e-4, "{c:?} {t:?}");
                assert!((c.amplitude - t.1).abs() < 1e-3 * t.1, "{c:?} {t:?}");
                assert!((c.sigma - t.2).abs() < 1e-4, "{c:?} {t:?}");
            }
        }
    }

    #[test]
    fn inflections_split_overlapping_echoes() {
        // 3 sigma apart: one smoothed maximum each? Here 2.2 sigma, where the sum has no dip.
        let y = gauss(100, &[(40.0, 100.0, 2.0), (44.4, 100.0, 2.0)], 0.0);
        let s = smooth(&y.iter().map(|&v| v as f64).collect::<Vec<_>>(), 0.0);
        let infl = find_peaks(&s, 1.0, PeakMethod::Inflection, 0.0);
        let der = find_peaks(&s, 1.0, PeakMethod::Derivative, 0.0);
        assert_eq!(infl.len(), 2);
        assert!(der.len() <= 2);
        let f = decompose_one(&y, 1.0, &DecomposeOptions { noise: Some(0.5), smooth: 0.0, ..Default::default() });
        assert_eq!(f.components.len(), 2);
    }

    #[test]
    fn flat_waveform_has_no_echo() {
        let y = vec![7.0f32; 50];
        let f = decompose_one(&y, 1.0, &DecomposeOptions { min_amplitude: 1.0, ..Default::default() });
        assert!(f.components.is_empty());
        assert_eq!(f.background, 7.0);
    }

    #[test]
    fn same_result_on_any_thread_count() {
        let mut wf = Waveforms::default();
        let mut g = crate::util::nprandom::Generator::new(1);
        for i in 0..300 {
            let y = gauss(80, &[(20.0 + (i % 7) as f64, 50.0, 2.0), (50.0, 30.0, 2.0)], 3.0);
            wf.pulse.push(i);
            wf.gps_time.push(0.0);
            wf.origin.push([0.0, 0.0, 0.0]);
            wf.anchor.push([0.0, 0.0, 0.0]);
            wf.direction.push([0.0, 0.0, 1.0]);
            wf.offset.push(0.0);
            wf.interval.push(1.0);
            wf.metres_per_ns.push(super::super::C_HALF);
            wf.sample_start.push(wf.samples.len());
            wf.sample_count.push(80);
            wf.samples.extend(y.iter().map(|&v| v + g.normal(0.0, 1.0) as f32));
        }
        let opts = DecomposeOptions::default();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| decompose(&wf, &opts).unwrap());
        let many = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| decompose(&wf, &opts).unwrap());
        assert_eq!(one, many);
        assert!(one.0.len() >= 550);
    }
}
