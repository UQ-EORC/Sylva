// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Stem maps as files: the JSON the Python package's `StemMap.save` writes,
//! read and written here so every binding shares one format, and the stem quality score.

use std::path::Path;

use crate::error::{Error, Result};
use crate::json::{self, Json};

/// One stem of a stem map (the Python package's `Stem`).
#[derive(Debug, Clone, PartialEq)]
pub struct StemRecord {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub dbh: f64,
    pub axis: [f64; 3],
    pub reference_height: f64,
    pub n_slices: i64,
    pub n_points: i64,
    pub rmse: f64,
    pub coverage: f64,
    pub lean_deg: f64,
}

impl StemRecord {
    /// A stem with the Python package's defaults for everything but its
    /// position and diameter.
    pub fn new(x: f64, y: f64, z: f64, dbh: f64) -> Self {
        StemRecord { x, y, z, dbh, axis: [0.0, 0.0, 1.0], reference_height: 1.3, n_slices: 0, n_points: 0, rmse: 0.0, coverage: 0.0, lean_deg: 0.0 }
    }

    /// 0-1 confidence from the fit residual, the arc seen and the slices linked.
    pub fn quality(&self) -> f64 {
        stem_quality(self.rmse, self.coverage, self.n_slices as f64)
    }
}

/// `clip(1 / (1 + rmse / 0.01) * coverage * min(n_slices / 6, 1), 0, 1)`.
pub fn stem_quality(rmse: f64, coverage: f64, n_slices: f64) -> f64 {
    let residual_term = 1.0 / (1.0 + rmse / 0.01);
    let slice_term = (n_slices / 6.0).min(1.0);
    let q = residual_term * coverage * slice_term;
    if q.is_nan() {
        q
    } else {
        q.clamp(0.0, 1.0)
    }
}

fn number(v: &Json, what: &str) -> Result<f64> {
    match v {
        Json::Int(i) => Ok(*i as f64),
        Json::Float(f) => Ok(*f),
        Json::Bool(b) => Ok(*b as i64 as f64),
        _ => Err(Error::invalid(format!("stem field `{what}` is not a number"))),
    }
}

fn stem_from_json(v: &Json) -> Result<StemRecord> {
    let req = |k: &str| v.get(k).ok_or_else(|| Error::invalid(format!("stem has no `{k}`"))).and_then(|x| number(x, k));
    let mut s = StemRecord::new(req("x")?, req("y")?, req("z")?, req("dbh")?);
    let opt = |k: &str, default: f64| v.get(k).map_or(Ok(default), |x| number(x, k));
    if let Some(a) = v.get("axis") {
        let Json::Array(items) = a else { return Err(Error::invalid("stem `axis` must be a list of three numbers")) };
        if items.len() != 3 {
            return Err(Error::invalid("stem `axis` must be a list of three numbers"));
        }
        for (k, item) in items.iter().enumerate() {
            s.axis[k] = number(item, "axis")?;
        }
    }
    s.reference_height = opt("reference_height", s.reference_height)?;
    s.n_slices = opt("n_slices", 0.0)? as i64;
    s.n_points = opt("n_points", 0.0)? as i64;
    s.rmse = opt("rmse", 0.0)?;
    s.coverage = opt("coverage", 0.0)?;
    s.lean_deg = opt("lean_deg", 0.0)?;
    Ok(s)
}

/// Stems from the text of a stem-map file: `(name, stems)`.
pub fn stem_map_from_json(text: &str) -> Result<(String, Vec<StemRecord>)> {
    let v = json::parse(text)?;
    let Some(Json::Array(stems)) = v.get("stems") else { return Err(Error::invalid("a stem map file needs a `stems` list")) };
    let name = match v.get("name") {
        Some(Json::Str(s)) => s.clone(),
        _ => String::new(),
    };
    Ok((name, stems.iter().map(stem_from_json).collect::<Result<_>>()?))
}

/// The text of a stem-map file, laid out as the Python package writes it.
pub fn stem_map_to_json(name: &str, stems: &[StemRecord]) -> String {
    let f = Json::Float;
    let items = stems
        .iter()
        .map(|s| {
            Json::Object(vec![
                ("x".into(), f(s.x)),
                ("y".into(), f(s.y)),
                ("z".into(), f(s.z)),
                ("dbh".into(), f(s.dbh)),
                ("axis".into(), Json::Array(s.axis.iter().map(|&a| f(a)).collect())),
                ("reference_height".into(), f(s.reference_height)),
                ("n_slices".into(), Json::Int(s.n_slices)),
                ("n_points".into(), Json::Int(s.n_points)),
                ("rmse".into(), f(s.rmse)),
                ("coverage".into(), f(s.coverage)),
                ("lean_deg".into(), f(s.lean_deg)),
            ])
        })
        .collect();
    json::to_string_indented(&Json::Object(vec![("name".into(), Json::Str(name.into())), ("stems".into(), Json::Array(items))]), 2)
}

/// Read a stem-map file.
pub fn read_stem_map(path: impl AsRef<Path>) -> Result<(String, Vec<StemRecord>)> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| Error::file(path, e.to_string()))?;
    stem_map_from_json(&text).map_err(|e| Error::file(path, e.to_string()))
}

/// Write a stem-map file, creating its directory.
pub fn write_stem_map(path: impl AsRef<Path>, name: &str, stems: &[StemRecord]) -> Result<()> {
    let path = path.as_ref();
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, stem_map_to_json(name, stems))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut s = StemRecord::new(1.5, -2.25, 0.1, 0.3);
        s.n_slices = 5;
        s.rmse = 0.005;
        s.coverage = 0.5;
        let text = stem_map_to_json("scan", &[s.clone()]);
        assert!(text.starts_with("{\n  \"name\": \"scan\",\n  \"stems\": [\n    {\n      \"x\": 1.5,"));
        let (name, back) = stem_map_from_json(&text).unwrap();
        assert_eq!(name, "scan");
        assert_eq!(back, vec![s.clone()]);
        assert!((s.quality() - 0.5 / 1.5 * 5.0 / 6.0).abs() < 1e-15);
        assert_eq!(stem_map_to_json("", &[]), "{\n  \"name\": \"\",\n  \"stems\": []\n}");
        assert!(stem_map_from_json("{\"stems\": [{\"x\": 1}]}").is_err());
    }
}
