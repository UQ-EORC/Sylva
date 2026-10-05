// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Did two epochs go through the same processing?
//!
//! A difference between epochs is only change if both were measured the
//! same way: a different stem-detection setting, voxel size or software
//! version can move a DBH or a canopy profile by more than a year's growth.
//! Each epoch's record (the Sylva version and the settings it was processed
//! with, as JSON) is flattened to dotted keys and the two are compared.

use crate::error::Result;
use crate::util::json::{self, py_float_repr, Json};

/// One setting that differs: its dotted key and the two values as text
/// (`None` where an epoch does not have it).
pub type Difference = (String, Option<String>, Option<String>);

fn text(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => py_float_repr(*f),
        Json::Str(s) => format!("{s:?}"),
        Json::Array(_) | Json::Object(_) => String::new(),
    }
}

fn flatten(prefix: &str, v: &Json, out: &mut Vec<(String, Json)>) {
    let key = |k: &str| if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") };
    match v {
        Json::Object(items) => {
            for (k, x) in items {
                flatten(&key(k), x, out);
            }
        }
        Json::Array(items) => {
            for (k, x) in items.iter().enumerate() {
                flatten(&key(&k.to_string()), x, out);
            }
            if items.is_empty() {
                out.push((prefix.to_string(), Json::Array(Vec::new())));
            }
        }
        _ => out.push((prefix.to_string(), v.clone())),
    }
}

fn same(a: &Json, b: &Json, rtol: f64) -> bool {
    let num = |v: &Json| match v {
        Json::Int(i) => Some(*i as f64),
        Json::Float(f) => Some(*f),
        _ => None,
    };
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= rtol * x.abs().max(y.abs()),
        _ => a == b,
    }
}

/// The settings that differ between two epoch records (JSON text), in the
/// order of the first record's keys, then the second's. Numbers within
/// `rtol` of each other count as equal.
pub fn differences(a: &str, b: &str, rtol: f64) -> Result<Vec<Difference>> {
    let (ja, jb) = (json::parse(a)?, json::parse(b)?);
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    flatten("", &ja, &mut fa);
    flatten("", &jb, &mut fb);
    let mut out = Vec::new();
    for (k, va) in &fa {
        match fb.iter().rev().find(|(kb, _)| kb == k) {
            Some((_, vb)) if same(va, vb, rtol) => {}
            Some((_, vb)) => out.push((k.clone(), Some(text(va)), Some(text(vb)))),
            None => out.push((k.clone(), Some(text(va)), None)),
        }
    }
    for (k, vb) in &fb {
        if !fa.iter().any(|(ka, _)| ka == k) {
            out.push((k.clone(), None, Some(text(vb))));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_what_differs() {
        let a = r#"{"sylva_version": "0.1.0", "settings": {"voxel": 0.01, "stems": {"min_slices": 3}, "hs": [1, 2]}}"#;
        let b = r#"{"sylva_version": "0.2.0", "settings": {"voxel": 0.010000000001, "stems": {"min_slices": 4}, "hs": [1, 2], "new": true}}"#;
        let d = differences(a, b, 1e-9).unwrap();
        let keys: Vec<&str> = d.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(keys, vec!["sylva_version", "settings.stems.min_slices", "settings.new"]);
        assert_eq!(d[1], ("settings.stems.min_slices".into(), Some("3".into()), Some("4".into())));
        assert_eq!(d[2].1, None);
        assert!(differences(a, a, 0.0).unwrap().is_empty());
        assert!(differences("{", a, 0.0).is_err());
    }
}
