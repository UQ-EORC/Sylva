// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Full-waveform lidar: a container of digitised waveforms, readers and
//! writers for LAS 1.3/1.4 waveform data packets ([`las`]) and PulseWaves
//! ([`pulsewaves`]), Gaussian decomposition ([`decompose`]) and a synthetic
//! waveform generator ([`simulate`]).
//!
//! A waveform is a run of samples of the received power, taken every
//! `interval` nanoseconds, along a straight beam. Its geometry is kept the
//! way both file formats keep it: an `anchor` point on the beam, the unit
//! `direction` of the beam (away from the scanner), the time `offset` (ns)
//! from the anchor to the first sample, and `metres_per_ns`, the range
//! travelled per nanosecond of round-trip time (half the speed of light in
//! the medium). Sample `i` then lies at
//! `anchor + direction * metres_per_ns * (offset + i * interval)`.
//! The pulse `origin` (the scanner's optical centre) is stored when the file
//! gives it, and is NaN otherwise.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::shots_ops::concatenate_attrs;
use crate::transform::{add, dot, normalize, scale, sub};
use crate::{Point, Shots};

pub mod decompose;
pub mod las;
pub mod pulsewaves;
pub mod simulate;

/// Range per nanosecond of round-trip time in vacuum, `c / 2` (m/ns).
pub const C_HALF: f64 = 0.299_792_458 / 2.0;

/// A set of digitised waveforms in compressed sparse row form: the samples
/// of waveform `i` are `samples[sample_start[i] .. sample_start[i] + sample_count[i]]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Waveforms {
    /// Pulse identifier; the rows of one pulse (segments, or outgoing and
    /// returning waveforms) are consecutive and share it.
    pub pulse: Vec<i64>,
    /// Time of the pulse (GPS time as stored in the file, s).
    pub gps_time: Vec<f64>,
    /// Optical centre of the scanner, NaN where the file does not give it.
    pub origin: Vec<Point>,
    /// Reference point on the beam.
    pub anchor: Vec<Point>,
    /// Unit beam direction, pointing away from the scanner.
    pub direction: Vec<Point>,
    /// Time from the anchor to the first sample (ns); negative when the
    /// first sample lies between the scanner and the anchor.
    pub offset: Vec<f64>,
    /// Sampling interval (ns).
    pub interval: Vec<f64>,
    /// Range per nanosecond of round-trip time (m/ns).
    pub metres_per_ns: Vec<f64>,
    pub sample_start: Vec<usize>,
    pub sample_count: Vec<u32>,
    pub samples: Vec<f32>,
    /// Per-waveform attributes (`kind`, `channel`, `intensity`, ...).
    pub attrs: BTreeMap<String, Attr>,
}

impl Waveforms {
    pub fn len(&self) -> usize {
        self.anchor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anchor.is_empty()
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Samples of waveform `i`.
    pub fn samples_of(&self, i: usize) -> &[f32] {
        let a = self.sample_start[i];
        &self.samples[a..a + self.sample_count[i] as usize]
    }

    /// Position at time `t` (ns after the first sample) along waveform `i`.
    pub fn position_at(&self, i: usize, t: f64) -> Point {
        add(&self.anchor[i], &scale(&self.direction[i], self.metres_per_ns[i] * (self.offset[i] + t)))
    }

    /// Position of every sample, in sample order.
    pub fn sample_positions(&self) -> Vec<Point> {
        let mut out = Vec::with_capacity(self.n_samples());
        for i in 0..self.len() {
            for k in 0..self.sample_count[i] {
                out.push(self.position_at(i, k as f64 * self.interval[i]));
            }
        }
        out
    }

    /// Time of every sample after its waveform's anchor (ns).
    pub fn sample_times(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_samples());
        for i in 0..self.len() {
            for k in 0..self.sample_count[i] {
                out.push(self.offset[i] + k as f64 * self.interval[i]);
            }
        }
        out
    }

    /// Check that the arrays agree in length and the CSR offsets are in range.
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        let lens = [
            ("pulse", self.pulse.len()), ("gps_time", self.gps_time.len()), ("origin", self.origin.len()),
            ("direction", self.direction.len()), ("offset", self.offset.len()), ("interval", self.interval.len()),
            ("metres_per_ns", self.metres_per_ns.len()), ("sample_start", self.sample_start.len()), ("sample_count", self.sample_count.len()),
        ];
        for (name, l) in lens {
            if l != n {
                return Err(Error::invalid(format!("waveforms: {name} has {l} values for {n} waveforms")));
            }
        }
        for (k, a) in &self.attrs {
            if a.len() != n {
                return Err(Error::invalid(format!("waveforms: attribute {k} has {} values for {n} waveforms", a.len())));
            }
        }
        for i in 0..n {
            let end = self.sample_start[i] + self.sample_count[i] as usize;
            if end > self.samples.len() {
                return Err(Error::invalid(format!("waveforms: waveform {i} runs past the end of the samples ({end} > {})", self.samples.len())));
            }
            if !(self.interval[i] > 0.0) || !self.interval[i].is_finite() {
                return Err(Error::invalid(format!("waveforms: interval of waveform {i} must be positive, got {}", self.interval[i])));
            }
            if !(self.metres_per_ns[i] > 0.0) || !self.metres_per_ns[i].is_finite() {
                return Err(Error::invalid(format!("waveforms: metres_per_ns of waveform {i} must be positive, got {}", self.metres_per_ns[i])));
            }
        }
        Ok(())
    }

    /// Keep the rows whose indices are in `idx` (in that order), with their samples.
    pub fn take(&self, idx: &[usize]) -> Waveforms {
        let mut out = Waveforms::default();
        for &i in idx {
            out.pulse.push(self.pulse[i]);
            out.gps_time.push(self.gps_time[i]);
            out.origin.push(self.origin[i]);
            out.anchor.push(self.anchor[i]);
            out.direction.push(self.direction[i]);
            out.offset.push(self.offset[i]);
            out.interval.push(self.interval[i]);
            out.metres_per_ns.push(self.metres_per_ns[i]);
            out.sample_start.push(out.samples.len());
            out.sample_count.push(self.sample_count[i]);
            out.samples.extend_from_slice(self.samples_of(i));
        }
        out.attrs = self.attrs.iter().map(|(k, v)| (k.clone(), v.take(idx))).collect();
        out
    }

    /// Stack waveform sets; attributes present in every part are kept.
    pub fn concatenate(parts: &[&Waveforms]) -> Result<Waveforms> {
        if parts.is_empty() {
            return Err(Error::invalid("no waveforms to concatenate"));
        }
        let mut out = Waveforms::default();
        for p in parts {
            out.pulse.extend_from_slice(&p.pulse);
            out.gps_time.extend_from_slice(&p.gps_time);
            out.origin.extend_from_slice(&p.origin);
            out.anchor.extend_from_slice(&p.anchor);
            out.direction.extend_from_slice(&p.direction);
            out.offset.extend_from_slice(&p.offset);
            out.interval.extend_from_slice(&p.interval);
            out.metres_per_ns.extend_from_slice(&p.metres_per_ns);
            let base = out.samples.len();
            out.sample_start.extend(p.sample_start.iter().map(|s| s + base));
            out.sample_count.extend_from_slice(&p.sample_count);
            out.samples.extend_from_slice(&p.samples);
        }
        for k in parts[0].attrs.keys() {
            if parts.iter().all(|p| p.attrs.contains_key(k)) {
                let cols: Vec<&Attr> = parts.iter().map(|p| &p.attrs[k]).collect();
                out.attrs.insert(k.clone(), concatenate_attrs(&cols));
            }
        }
        Ok(out)
    }

    /// Row ranges of the pulses: consecutive rows with equal `pulse`.
    pub fn pulse_groups(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.len() {
            let mut j = i + 1;
            while j < self.len() && self.pulse[j] == self.pulse[i] {
                j += 1;
            }
            out.push((i, j));
            i = j;
        }
        out
    }

    /// Per-row attribute `kind` as integers (0 unknown, 1 outgoing,
    /// 2 returning); every row is returning when the attribute is absent.
    pub fn kinds(&self) -> Vec<i64> {
        match self.attrs.get("kind") {
            Some(a) => a.to_f64().iter().map(|&v| v as i64).collect(),
            None => vec![2; self.len()],
        }
    }
}

/// Echoes found in waveforms: one row per echo, grouped by waveform.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Echoes {
    /// Row of the waveform each echo came from, ascending.
    pub waveform: Vec<usize>,
    /// Time of the echo after the waveform's first sample (ns).
    pub time: Vec<f64>,
    /// Peak amplitude above the background (sample units).
    pub amplitude: Vec<f64>,
    /// Standard deviation of the fitted Gaussian (ns).
    pub width: Vec<f64>,
    /// Echo position.
    pub xyz: Vec<Point>,
    /// Range from the pulse origin along the beam (m); NaN without an origin.
    pub range: Vec<f64>,
}

impl Echoes {
    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// Waveform file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// LAS 1.3 / 1.4 (or LAZ) with waveform data packets.
    Las,
    /// PulseWaves `.pls` with its `.wvs`.
    PulseWaves,
}

/// Tell the format from the file's signature.
pub fn detect_format(path: impl AsRef<std::path::Path>) -> Result<FileFormat> {
    use std::io::Read;
    let path = path.as_ref();
    let mut f = std::fs::File::open(path).map_err(|e| Error::file(path, e.to_string()))?;
    let mut sig = [0u8; 15];
    let mut got = 0;
    while got < sig.len() {
        let n = f.read(&mut sig[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    if got >= 4 && &sig[..4] == b"LASF" {
        Ok(FileFormat::Las)
    } else if got == 15 && &sig == b"PulseWavesPulse" {
        Ok(FileFormat::PulseWaves)
    } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("sdf")) {
        Err(Error::file(path, "RIEGL SDF files can only be read with RIEGL's proprietary library; export the waveforms with RIEGL software (RiPROCESS or RiANALYZE) to LAS 1.3 or 1.4 with waveform data packets and read that"))
    } else {
        Err(Error::file(path, "not a waveform file: expected LAS/LAZ with waveform data packets or PulseWaves .pls"))
    }
}

/// Backscatter cross-section of Wagner et al. (2006):
/// `sigma = C_cal R^4 P s`, with `P` the echo amplitude, `s` its width and
/// `R` the range. With `calibration = 1` the values are relative.
pub fn backscatter_cross_section(range: &[f64], amplitude: &[f64], width: &[f64], calibration: f64) -> Vec<f64> {
    range.iter().zip(amplitude).zip(width).map(|((&r, &a), &s)| calibration * r.powi(4) * a * s).collect()
}

/// Calibration constant from echoes of extended Lambertian targets of known
/// reflectance (Wagner 2010): for such a target
/// `sigma = pi rho R^2 beta^2 cos(alpha)`, so each echo gives
/// `C_cal = pi rho beta^2 cos(alpha) / (R^2 P s)`; the median is returned.
pub fn calibration_constant(range: &[f64], amplitude: &[f64], width: &[f64], reflectance: &[f64], beam_divergence: f64, incidence: &[f64]) -> Result<f64> {
    let n = range.len();
    if amplitude.len() != n || width.len() != n || reflectance.len() != n || incidence.len() != n {
        return Err(Error::invalid("calibration: range, amplitude, width, reflectance and incidence must have the same length"));
    }
    let mut c: Vec<f64> = (0..n)
        .map(|i| std::f64::consts::PI * reflectance[i] * beam_divergence * beam_divergence * incidence[i].cos() / (range[i] * range[i] * amplitude[i] * width[i]))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if c.is_empty() {
        return Err(Error::invalid("calibration: no echo with a finite, positive range, amplitude and width"));
    }
    c.sort_by(|a, b| a.total_cmp(b));
    let m = c.len();
    Ok(if m % 2 == 1 { c[m / 2] } else { 0.5 * (c[m / 2 - 1] + c[m / 2]) })
}

/// Build pulses from waveforms and the echoes found in them: one shot per
/// pulse (consecutive rows sharing `pulse`; outgoing rows are skipped), a
/// pulse without an echo is kept as a miss.
///
/// The shot starts at `origin[row]` when given, else at the waveform's
/// origin, else (the origin unknown) at the position of the first sample of
/// the pulse's first returning waveform. Echo ranges are measured from that
/// start along the beam and sorted. Echo attributes: `amplitude`, `width`
/// (ns), `time` (ns after the first sample of its waveform) and `waveform`
/// (row).
pub fn to_shots(wf: &Waveforms, echoes: &Echoes, origin: Option<&[Point]>) -> Result<Shots> {
    if let Some(o) = origin {
        if o.len() != wf.len() {
            return Err(Error::invalid(format!("origin has {} rows for {} waveforms", o.len(), wf.len())));
        }
    }
    let n_echo = echoes.len();
    if echoes.waveform.iter().any(|&w| w >= wf.len()) {
        return Err(Error::invalid("echoes refer to waveforms that are not in the set"));
    }
    // Echo rows per waveform (echoes are grouped by waveform, ascending).
    let mut first = vec![usize::MAX; wf.len() + 1];
    for e in (0..n_echo).rev() {
        first[echoes.waveform[e]] = e;
    }
    let kinds = wf.kinds();
    let mut shots = Shots::default();
    let (mut amp, mut width, mut time, mut row) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (a, b) in wf.pulse_groups() {
        let rows: Vec<usize> = (a..b).filter(|&r| kinds[r] != 1).collect();
        if rows.is_empty() {
            continue;
        }
        let r0 = rows[0];
        let dir = normalize(&wf.direction[r0]);
        let start = match origin {
            Some(o) if o[r0].iter().all(|v| v.is_finite()) => o[r0],
            _ if wf.origin[r0].iter().all(|v| v.is_finite()) => wf.origin[r0],
            _ => wf.position_at(r0, 0.0),
        };
        let mut members: Vec<(f64, usize)> = Vec::new();
        for &r in &rows {
            if first[r] == usize::MAX {
                continue;
            }
            let mut e = first[r];
            while e < n_echo && echoes.waveform[e] == r {
                members.push((dot(&sub(&echoes.xyz[e], &start), &dir), e));
                e += 1;
            }
        }
        members.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        shots.origin.push(start);
        shots.direction.push(dir);
        shots.echo_start.push(shots.echo_range.len());
        shots.echo_count.push(members.len() as u32);
        for (rng, e) in members {
            shots.echo_range.push(rng);
            amp.push(echoes.amplitude[e]);
            width.push(echoes.width[e]);
            time.push(echoes.time[e]);
            row.push(echoes.waveform[e] as i64);
        }
    }
    shots.echo_attrs.insert("amplitude".into(), Attr::F64(amp));
    shots.echo_attrs.insert("width".into(), Attr::F64(width));
    shots.echo_attrs.insert("time".into(), Attr::F64(time));
    shots.echo_attrs.insert("waveform".into(), Attr::I64(row));
    Ok(shots)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn two_waveforms() -> Waveforms {
        let mut w = Waveforms {
            pulse: vec![0, 1],
            gps_time: vec![1.0, 2.0],
            origin: vec![[0.0, 0.0, 100.0], [f64::NAN; 3]],
            anchor: vec![[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]],
            direction: vec![[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            offset: vec![-10.0, 0.0],
            interval: vec![1.0, 2.0],
            metres_per_ns: vec![C_HALF, C_HALF],
            sample_start: vec![0, 4],
            sample_count: vec![4, 3],
            samples: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            attrs: BTreeMap::new(),
        };
        w.attrs.insert("intensity".into(), Attr::U16(vec![10, 20]));
        w
    }

    #[test]
    fn positions_follow_the_beam() {
        let w = two_waveforms();
        w.validate().unwrap();
        let p = w.sample_positions();
        assert_eq!(p.len(), 7);
        assert!((p[0][2] - (10.0 + 10.0 * C_HALF)).abs() < 1e-12);
        assert!((p[3][2] - (10.0 + 7.0 * C_HALF)).abs() < 1e-12);
        assert!((p[6][2] - (10.0 - 4.0 * C_HALF)).abs() < 1e-12);
        assert_eq!(w.sample_times()[6], 4.0);
    }

    #[test]
    fn take_and_concatenate() {
        let w = two_waveforms();
        let t = w.take(&[1]);
        assert_eq!(t.samples, vec![5.0, 6.0, 7.0]);
        assert_eq!(t.sample_start, vec![0]);
        let c = Waveforms::concatenate(&[&w, &t]).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.samples_of(2), &[5.0, 6.0, 7.0]);
        c.validate().unwrap();
        let mut bad = w.clone();
        bad.sample_count[1] = 9;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn shots_from_echoes() {
        let w = two_waveforms();
        let e = Echoes {
            waveform: vec![0, 0],
            time: vec![3.0, 1.0],
            amplitude: vec![1.0, 2.0],
            width: vec![1.0, 1.0],
            xyz: vec![w.position_at(0, 3.0), w.position_at(0, 1.0)],
            range: vec![f64::NAN; 2],
        };
        let s = to_shots(&w, &e, None).unwrap();
        assert_eq!(s.n_shots(), 2);
        assert_eq!(s.echo_count, vec![2, 0]);
        // From the origin at z = 100 to z = 10 + 9 C_HALF (t = 1) and 10 + 7 C_HALF (t = 3).
        assert!((s.echo_range[0] - (90.0 - 9.0 * C_HALF)).abs() < 1e-9);
        assert!((s.echo_range[1] - (90.0 - 7.0 * C_HALF)).abs() < 1e-9);
        // The second pulse has no origin: it starts at its first sample.
        assert!((s.origin[1][2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn cross_section_calibration_inverts() {
        let (r, a, s) = (vec![100.0, 200.0], vec![50.0, 12.5], vec![2.0, 2.0]);
        let c = calibration_constant(&r, &a, &s, &[0.5, 0.5], 0.0005, &[0.0, 0.0]).unwrap();
        let sigma = backscatter_cross_section(&r, &a, &s, c);
        let expect = |rr: f64| std::f64::consts::PI * 0.5 * rr * rr * 0.0005f64.powi(2);
        assert!((sigma[0] / expect(100.0) - 1.0).abs() < 1e-12);
        assert!((sigma[1] / expect(200.0) - 1.0).abs() < 1e-12);
        let e = calibration_constant(&r, &a[..1], &s, &[0.5, 0.5], 0.0005, &[0.0, 0.0]).unwrap_err().to_string();
        assert_eq!(e, "calibration: range, amplitude, width, reflectance and incidence must have the same length");
    }

    #[test]
    fn malformed_sets_say_what_is_wrong() {
        let err = |f: &dyn Fn(&mut Waveforms)| {
            let mut w = two_waveforms();
            f(&mut w);
            w.validate().unwrap_err().to_string()
        };
        assert_eq!(err(&|w| w.gps_time.pop().map(drop).unwrap()), "waveforms: gps_time has 1 values for 2 waveforms");
        assert_eq!(err(&|w| drop(w.attrs.insert("intensity".into(), Attr::U16(vec![1])))), "waveforms: attribute intensity has 1 values for 2 waveforms");
        assert_eq!(err(&|w| w.interval[1] = 0.0), "waveforms: interval of waveform 1 must be positive, got 0");
        assert_eq!(err(&|w| w.metres_per_ns[0] = f64::INFINITY), "waveforms: metres_per_ns of waveform 0 must be positive, got inf");
        assert!(Waveforms::default().is_empty() && !two_waveforms().is_empty());
        assert_eq!(Waveforms::concatenate(&[]).unwrap_err().to_string(), "no waveforms to concatenate");
    }

    #[test]
    fn outgoing_rows_and_foreign_echoes() {
        // Pulse 0 has an outgoing (kind 1) and a returning row; the outgoing one is skipped.
        let mut w = two_waveforms();
        w.pulse = vec![0, 0];
        w.attrs.insert("kind".into(), Attr::U8(vec![1, 2]));
        let none = Echoes::default();
        assert!(none.is_empty());
        let s = to_shots(&w, &none, None).unwrap();
        assert_eq!((s.n_shots(), s.echo_count.clone()), (1, vec![0]));
        assert_eq!(s.origin[0], w.position_at(1, 0.0), "the returning row's first sample");
        w.attrs.insert("kind".into(), Attr::U8(vec![1, 1]));
        assert_eq!(to_shots(&w, &none, None).unwrap().n_shots(), 0, "a pulse with only outgoing rows");
        let stray = Echoes { waveform: vec![5], time: vec![0.0], amplitude: vec![1.0], width: vec![1.0], xyz: vec![[0.0; 3]], range: vec![f64::NAN] };
        assert_eq!(to_shots(&w, &stray, None).unwrap_err().to_string(), "echoes refer to waveforms that are not in the set");
        assert_eq!(to_shots(&w, &none, Some(&[[0.0; 3]])).unwrap_err().to_string(), "origin has 1 rows for 2 waveforms");
    }

    #[test]
    fn formats_are_told_by_their_signature() {
        let dir = std::env::temp_dir().join(format!("sylva-wf-detect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        };
        assert_eq!(detect_format(write("a.bin", b"LASF\x01\x02")).unwrap(), FileFormat::Las);
        assert_eq!(detect_format(write("a.pls", b"PulseWavesPulse\x00")).unwrap(), FileFormat::PulseWaves);
        assert!(detect_format(write("a.sdf", b"RIEGL")).unwrap_err().to_string().contains("RIEGL SDF files can only be read with RIEGL's proprietary library"));
        let short = write("short.las", b"LA");
        assert_eq!(detect_format(&short).unwrap_err().to_string(), format!("{}: not a waveform file: expected LAS/LAZ with waveform data packets or PulseWaves .pls", short.display()));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(detect_format(dir.join("gone.las")).unwrap_err().to_string().starts_with(&format!("{}: ", dir.join("gone.las").display())));
    }
}
