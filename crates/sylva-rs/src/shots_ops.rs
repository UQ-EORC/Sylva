// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Operations on [`Shots`] beyond the container: stacking pulse sets, beam
//! angles, and the pulses a RIEGL point stream leaves out.
//!
//! RiVLib's stream only holds pulses that returned something; given the
//! angular scan pattern, [`Shots::fill_missing`] adds the rest as echo-less
//! shots with random azimuths, drawn from NumPy's generator so that a seed
//! gives the misses the Python package always gave.

use std::collections::BTreeMap;

use crate::canopy_profile::{pulses_per_line, zenith_deg, ScanPattern};
use crate::error::{Error, Result};
use crate::nprandom::Generator;
use crate::numeric::histogram;
use crate::pointcloud::Attr;
use crate::shots::Shots;

/// `np.remainder(a, b)` for floats: the result takes the sign of `b`, and a
/// zero result is `+0.0` for positive `b`.
pub fn np_remainder(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// Azimuth (degrees clockwise from +y, in `[0, 360)`) of each direction.
pub fn azimuth_deg(direction: &[crate::Point]) -> Vec<f64> {
    direction.iter().map(|d| np_remainder(d[0].atan2(d[1]) * (180.0 / std::f64::consts::PI), 360.0)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Bool,
    Signed(u32),
    Unsigned(u32),
    Float(u32),
}

fn kind(a: &Attr) -> Kind {
    match a {
        Attr::Bool(_) => Kind::Bool,
        Attr::I8(_) => Kind::Signed(8),
        Attr::I32(_) => Kind::Signed(32),
        Attr::I64(_) => Kind::Signed(64),
        Attr::U8(_) => Kind::Unsigned(8),
        Attr::U16(_) => Kind::Unsigned(16),
        Attr::U32(_) => Kind::Unsigned(32),
        Attr::F32(_) => Kind::Float(32),
        Attr::F64(_) => Kind::Float(64),
    }
}

/// NumPy's type promotion over the attribute types (a 16-bit signed result,
/// which has no column type, widens to 32 bits).
fn promote(a: Kind, b: Kind) -> Kind {
    use Kind::*;
    match (a, b) {
        _ if a == b => a,
        (Bool, k) | (k, Bool) => k,
        (Float(x), Float(y)) => Float(x.max(y)),
        (Float(f), Signed(i) | Unsigned(i)) | (Signed(i) | Unsigned(i), Float(f)) => Float(if f == 64 || i >= 32 { 64 } else { 32 }),
        (Signed(x), Signed(y)) => Signed(x.max(y)),
        (Unsigned(x), Unsigned(y)) => Unsigned(x.max(y)),
        (Signed(i), Unsigned(u)) | (Unsigned(u), Signed(i)) => Signed(if u < i { i } else { (2 * u).max(32) }),
    }
}

fn integers(a: &Attr) -> Vec<i64> {
    match a {
        Attr::Bool(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::I8(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::I32(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::I64(v) => v.clone(),
        Attr::U8(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::U16(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::U32(v) => v.iter().map(|&x| x as i64).collect(),
        Attr::F32(_) | Attr::F64(_) => unreachable!("floats are not cast to integers"),
    }
}

/// `np.concatenate` of attribute columns, promoting mixed types as NumPy does.
pub fn concatenate_attrs(cols: &[&Attr]) -> Attr {
    let target = cols.iter().map(|a| kind(a)).reduce(promote).unwrap_or(Kind::Float(64));
    if cols.iter().all(|a| kind(a) == target) {
        let mut out = cols[0].clone();
        for c in &cols[1..] {
            out.extend(c).expect("same type");
        }
        return out;
    }
    match target {
        Kind::Float(64) => Attr::F64(cols.iter().flat_map(|a| a.to_f64()).collect()),
        Kind::Float(_) => Attr::F32(cols.iter().flat_map(|a| a.to_f64()).map(|v| v as f32).collect()),
        k => {
            let v: Vec<i64> = cols.iter().flat_map(|a| integers(a)).collect();
            match k {
                Kind::Signed(8) => Attr::I8(v.into_iter().map(|x| x as i8).collect()),
                Kind::Signed(32) => Attr::I32(v.into_iter().map(|x| x as i32).collect()),
                Kind::Signed(_) => Attr::I64(v),
                Kind::Unsigned(8) => Attr::U8(v.into_iter().map(|x| x as u8).collect()),
                Kind::Unsigned(16) => Attr::U16(v.into_iter().map(|x| x as u16).collect()),
                Kind::Unsigned(_) => Attr::U32(v.into_iter().map(|x| x as u32).collect()),
                Kind::Bool | Kind::Float(_) => unreachable!(),
            }
        }
    }
}

/// CSR offsets of packed echoes: `[0, cumsum(count)[:-1]]`.
pub fn packed_starts(count: &[u32]) -> Vec<usize> {
    let mut acc = 0usize;
    count
        .iter()
        .map(|&c| {
            let s = acc;
            acc += c as usize;
            s
        })
        .collect()
}

impl Shots {
    /// Zenith (degrees from +z) and azimuth (degrees clockwise from +y, in
    /// `[0, 360)`, RIEGL's convention) of each pulse.
    pub fn zenith_azimuth(&self) -> (Vec<f64>, Vec<f64>) {
        (zenith_deg(&self.direction), azimuth_deg(&self.direction))
    }

    /// Stack pulse sets in order. Echoes are taken in each part's order and
    /// the offsets rebuilt; only echo attributes present in every part are
    /// kept, promoted to a common type where the parts disagree.
    pub fn concatenate(parts: &[&Shots]) -> Result<Shots> {
        if parts.is_empty() {
            return Err(Error::invalid("no shots to concatenate"));
        }
        let mut out = Shots::default();
        for p in parts {
            out.origin.extend_from_slice(&p.origin);
            out.direction.extend_from_slice(&p.direction);
            out.echo_count.extend_from_slice(&p.echo_count);
            out.echo_range.extend_from_slice(&p.echo_range);
        }
        out.echo_start = packed_starts(&out.echo_count);
        out.echo_attrs = parts[0]
            .echo_attrs
            .keys()
            .filter(|k| parts.iter().all(|p| p.echo_attrs.contains_key(*k)))
            .map(|k| (k.clone(), concatenate_attrs(&parts.iter().map(|p| &p.echo_attrs[k]).collect::<Vec<_>>())))
            .collect::<BTreeMap<_, _>>();
        Ok(out)
    }

    /// The pulses that returned nothing, from the scan `pattern`: per zenith
    /// line, the pulses fired (`per_line`, else estimated by
    /// [`pulses_per_line`] at the 0.98 quantile) less the shots observed,
    /// as echo-less shots from the mean origin with azimuths drawn
    /// uniformly by `np.random.default_rng(seed)`. `None` when nothing is
    /// missing; otherwise the input followed by the misses. Call in the
    /// scanner frame, where the pattern's zenith lines are defined.
    pub fn fill_missing(&self, pattern: &ScanPattern, per_line: Option<i64>, seed: u64, shot_stride: usize) -> Option<Shots> {
        let theta = pattern.lines();
        let zen = zenith_deg(&self.direction);
        let observed = histogram(zen.iter().copied(), &pattern.line_edges());
        let per_line = per_line.unwrap_or_else(|| pulses_per_line(&zen, pattern, 0.98, shot_stride) as i64);
        let missing: Vec<usize> = observed.iter().map(|&o| (per_line - o as i64).max(0) as usize).collect();
        let n: usize = missing.iter().sum();
        if n == 0 {
            return None;
        }
        let az = Generator::new(seed).uniform_n(0.0, 360.0, n);
        let to_rad = std::f64::consts::PI / 180.0;
        let zen_new = theta.iter().zip(&missing).flat_map(|(&t, &m)| std::iter::repeat_n(t * to_rad, m));
        let direction: Vec<crate::Point> = zen_new
            .zip(&az)
            .map(|(z, &a)| {
                let a = a * to_rad;
                [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
            })
            .collect();
        // origin.mean(axis=0): a running sum down each column.
        let mut sum = [0.0; 3];
        if let Some(first) = self.origin.first() {
            sum = *first;
            for o in &self.origin[1..] {
                for k in 0..3 {
                    sum[k] += o[k];
                }
            }
        }
        let count = self.origin.len() as f64;
        let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
        let empty = Shots {
            origin: vec![mean; n],
            direction,
            echo_start: vec![0; n],
            echo_count: vec![0; n],
            echo_range: Vec::new(),
            echo_attrs: self.echo_attrs.iter().map(|(k, v)| (k.clone(), v.take(&[]))).collect(),
        };
        Some(Shots::concatenate(&[self, &empty]).expect("two parts"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shots(counts: &[u32]) -> Shots {
        let n = counts.len();
        let m: usize = counts.iter().map(|&c| c as usize).sum();
        let mut attrs = BTreeMap::new();
        attrs.insert("a".to_string(), Attr::U8((0..m as u8).collect()));
        Shots {
            origin: vec![[0.0, 0.0, 1.0]; n],
            direction: vec![[0.0, 0.0, 1.0]; n],
            echo_start: packed_starts(counts),
            echo_count: counts.to_vec(),
            echo_range: (0..m).map(|i| i as f64 + 1.0).collect(),
            echo_attrs: attrs,
        }
    }

    #[test]
    fn remainder_follows_numpy() {
        assert_eq!(np_remainder(-30.0, 360.0), 330.0);
        assert_eq!(np_remainder(-0.0, 360.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(np_remainder(-1e-20, 360.0), 360.0);
        assert_eq!(np_remainder(725.0, 360.0), 5.0);
    }

    #[test]
    fn concatenate_rebuilds_offsets_and_promotes() {
        let a = shots(&[2, 0, 1]);
        let mut b = shots(&[0, 3]);
        b.echo_attrs.insert("a".into(), Attr::I8(vec![-1, -2, -3]));
        b.echo_attrs.insert("only_b".into(), Attr::F64(vec![0.0; 3]));
        let c = Shots::concatenate(&[&a, &b]).unwrap();
        assert_eq!(c.echo_start, vec![0, 2, 2, 3, 3]);
        assert_eq!(c.echo_attrs.keys().collect::<Vec<_>>(), vec!["a"]);
        // uint8 with int8 is int16 in NumPy; the narrowest column type is int32.
        assert_eq!(c.echo_attrs["a"], Attr::I32(vec![0, 1, 2, -1, -2, -3]));
        assert!(Shots::concatenate(&[]).is_err());
        assert_eq!(promote(Kind::Float(32), Kind::Unsigned(16)), Kind::Float(32));
        assert_eq!(promote(Kind::Float(32), Kind::Signed(32)), Kind::Float(64));
        assert_eq!(promote(Kind::Unsigned(32), Kind::Signed(32)), Kind::Signed(64));
        assert_eq!(promote(Kind::Unsigned(8), Kind::Signed(32)), Kind::Signed(32));
    }

    #[test]
    fn fill_missing_tops_up_each_line() {
        let pattern = ScanPattern { theta_start: 80.0, theta_delta: 10.0, theta_count: 3, phi_count: 4 };
        let mut s = shots(&[1, 1, 1]);
        s.direction = [80.0f64, 90.0, 90.0].iter().map(|t| [t.to_radians().sin(), 0.0, t.to_radians().cos()]).collect();
        let f = s.fill_missing(&pattern, None, 0, 1).unwrap();
        assert_eq!(f.n_shots(), 12);
        let (zen, _) = f.zenith_azimuth();
        assert_eq!(histogram(zen.iter().copied(), &pattern.line_edges()), vec![4, 4, 4]);
        assert!(s.fill_missing(&pattern, Some(0), 0, 1).is_none());
        assert_eq!(s.fill_missing(&pattern, Some(1), 0, 1).unwrap().n_shots(), 4);
    }
}
