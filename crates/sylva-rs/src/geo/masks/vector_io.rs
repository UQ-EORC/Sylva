// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Reading polygon layers from ESRI shapefiles and GeoJSON.
//!
//! Each record (a shapefile shape or a GeoJSON feature) becomes one
//! [`Feature`]: a [`MultiPolygon`] and its attribute table row. Coordinates
//! are read as they are stored; the layer's CRS (the `.prj` WKT of a
//! shapefile, or the legacy `crs` member of a GeoJSON file) is returned as
//! text and never applied.

use std::path::{Path, PathBuf};

use geojson::{GeoJson, Geometry, GeometryValue, JsonValue};
use shapefile::dbase;

use super::polygon::{ring_contains, MultiPolygon, Polygon, Xy};
use crate::error::{Error, Result};

/// One attribute value of a feature.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// One record of a polygon layer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Feature {
    pub geometry: MultiPolygon,
    /// Attribute name and value: in field order for a shapefile, sorted by
    /// name for GeoJSON.
    pub properties: Vec<(String, Property)>,
}

/// A polygon layer read from a file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layer {
    pub features: Vec<Feature>,
    /// CRS as stored in the file (WKT or a name such as `EPSG:28355`).
    pub crs: Option<String>,
}

fn is_ext(p: &Path, exts: &[&str]) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

const SHP: &[&str] = &["shp"];
const GEOJSON: &[&str] = &["geojson", "json"];

/// Resolve `path` and `layer` to one file.
///
/// A file is read as it is (`layer`, if given, must equal its stem). A
/// directory is searched for `<layer>.shp`, `<layer>.geojson` or
/// `<layer>.json`; without `layer` it must hold exactly one such file.
fn resolve(path: &Path, layer: Option<&str>) -> Result<PathBuf> {
    if path.is_dir() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(path)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && (is_ext(p, SHP) || is_ext(p, GEOJSON))).collect();
        found.sort();
        let names = || found.iter().filter_map(|p| p.file_name().and_then(|n| n.to_str())).collect::<Vec<_>>().join(", ");
        return match layer {
            Some(l) => {
                let hits: Vec<&PathBuf> = found.iter().filter(|p| p.file_stem().and_then(|s| s.to_str()) == Some(l)).collect();
                match hits.as_slice() {
                    [one] => Ok((*one).clone()),
                    [] => Err(Error::file(path, format!("no layer {l:?}; the directory holds: {}", if found.is_empty() { "no shapefiles or GeoJSON files".into() } else { names() }))),
                    _ => Err(Error::file(path, format!("layer {l:?} is ambiguous: {}", hits.iter().filter_map(|p| p.file_name().and_then(|n| n.to_str())).collect::<Vec<_>>().join(", ")))),
                }
            }
            None => match found.as_slice() {
                [one] => Ok(one.clone()),
                [] => Err(Error::file(path, "the directory holds no shapefiles or GeoJSON files")),
                _ => Err(Error::file(path, format!("the directory holds several layers; choose one with layer=: {}", names()))),
            },
        };
    }
    if !path.exists() {
        return Err(Error::file(path, "no such file or directory"));
    }
    if let Some(l) = layer {
        if path.file_stem().and_then(|s| s.to_str()) != Some(l) {
            return Err(Error::file(path, format!("layer {l:?} does not match this file; a layer selects a file within a directory")));
        }
    }
    Ok(path.to_path_buf())
}

/// Read the polygons of a shapefile (`.shp`, with its `.dbf` and `.prj` if
/// present) or a GeoJSON file (`.geojson`, `.json`).
///
/// `path` may also be a directory, in which case `layer` names the file
/// (without extension) to read.
///
/// Null geometries give features with no parts. Any other non-polygon
/// geometry is an error, so that a layer of lines or points is not silently
/// read as empty.
pub fn read_polygons(path: impl AsRef<Path>, layer: Option<&str>) -> Result<Layer> {
    let file = resolve(path.as_ref(), layer)?;
    if is_ext(&file, SHP) {
        read_shapefile(&file)
    } else if is_ext(&file, GEOJSON) {
        read_geojson(&file)
    } else {
        Err(Error::UnsupportedFormat(format!("{}: polygons are read from .shp, .geojson or .json files", file.display())))
    }
}

// ------------------------------------------------------------------ shapefile

fn read_shapefile(path: &Path) -> Result<Layer> {
    let ferr = |e: shapefile::Error| Error::file(path, e.to_string());
    let reader = shapefile::ShapeReader::from_path(path).map_err(ferr)?;
    let shapes = reader.read().map_err(ferr)?;
    let mut features = Vec::with_capacity(shapes.len());
    for (i, shape) in shapes.into_iter().enumerate() {
        use shapefile::Shape;
        let rings: Vec<(bool, Vec<Xy>)> = match shape {
            Shape::NullShape => Vec::new(),
            Shape::Polygon(p) => p.into_inner().into_iter().map(|r| (matches!(r, shapefile::PolygonRing::Inner(_)), r.into_inner().into_iter().map(|q| [q.x, q.y]).collect())).collect(),
            Shape::PolygonM(p) => p.into_inner().into_iter().map(|r| (matches!(r, shapefile::PolygonRing::Inner(_)), r.into_inner().into_iter().map(|q| [q.x, q.y]).collect())).collect(),
            Shape::PolygonZ(p) => p.into_inner().into_iter().map(|r| (matches!(r, shapefile::PolygonRing::Inner(_)), r.into_inner().into_iter().map(|q| [q.x, q.y]).collect())).collect(),
            other => return Err(Error::file(path, format!("shape {i} is a {}, not a polygon", other.shapetype()))),
        };
        features.push(Feature { geometry: group_rings(rings), properties: Vec::new() });
    }
    let dbf = path.with_extension("dbf");
    if dbf.is_file() {
        let derr = |e: dbase::Error| Error::file(&dbf, e.to_string());
        let mut reader = dbase::ReaderBuilder::new().with_encoding(dbase::UnicodeLossy).open(&dbf).map_err(derr)?;
        let names: Vec<String> = reader.fields().iter().map(|f| f.name().to_string()).collect();
        let records = reader.read().map_err(derr)?;
        if records.len() != features.len() {
            return Err(Error::file(&dbf, format!("{} records for {} shapes", records.len(), features.len())));
        }
        for (feature, mut record) in features.iter_mut().zip(records) {
            feature.properties = names.iter().map(|n| (n.clone(), record.remove(n).map_or(Property::Null, dbase_value))).collect();
        }
    }
    let prj = path.with_extension("prj");
    let crs = if prj.is_file() { Some(std::fs::read_to_string(&prj)?.trim().to_string()).filter(|s| !s.is_empty()) } else { None };
    Ok(Layer { features, crs })
}

/// Shapefile rings in file order to polygons: each exterior ring starts a
/// polygon and each hole joins the latest exterior ring that contains it
/// (the latest one, if none does). A hole before any exterior ring is read
/// as an exterior ring.
fn group_rings(rings: Vec<(bool, Vec<Xy>)>) -> MultiPolygon {
    let mut parts: Vec<Polygon> = Vec::new();
    for (hole, ring) in rings {
        if !hole || parts.is_empty() {
            parts.push(Polygon { exterior: ring, holes: Vec::new() });
            continue;
        }
        let target = ring.first().and_then(|&p| parts.iter().rposition(|part| ring_contains(&part.exterior, p))).unwrap_or(parts.len() - 1);
        parts[target].holes.push(ring);
    }
    MultiPolygon { parts }
}

fn dbase_value(v: dbase::FieldValue) -> Property {
    use dbase::FieldValue as F;
    match v {
        F::Character(s) => s.map_or(Property::Null, Property::Str),
        F::Numeric(x) => x.map_or(Property::Null, number),
        F::Float(x) => x.map_or(Property::Null, |x| number(x as f64)),
        F::Logical(b) => b.map_or(Property::Null, Property::Bool),
        F::Date(d) => d.map_or(Property::Null, |d| Property::Str(format!("{:04}-{:02}-{:02}", d.year(), d.month(), d.day()))),
        F::Integer(i) => Property::Int(i as i64),
        F::Currency(x) | F::Double(x) => Property::Float(x),
        F::DateTime(dt) => {
            let (d, t) = (dt.date(), dt.time());
            Property::Str(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", d.year(), d.month(), d.day(), t.hours(), t.minutes(), t.seconds()))
        }
        F::Memo(s) => Property::Str(s),
    }
}

/// dBase numeric fields hold both integers and decimals; keep whole numbers
/// as integers so that identifiers stay integers.
fn number(x: f64) -> Property {
    if x.fract() == 0.0 && x.abs() < 9.007_199_254_740_992e15 {
        Property::Int(x as i64)
    } else {
        Property::Float(x)
    }
}

// -------------------------------------------------------------------- GeoJSON

fn read_geojson(path: &Path) -> Result<Layer> {
    let text = std::fs::read_to_string(path)?;
    let gj: GeoJson = text.parse().map_err(|e: geojson::Error| Error::file(path, format!("invalid GeoJSON: {e}")))?;
    let mut crs = None;
    let features = match gj {
        GeoJson::Geometry(g) => vec![Feature { geometry: geometry_polygons(path, 0, &g)?, properties: Vec::new() }],
        GeoJson::Feature(f) => vec![geojson_feature(path, 0, f)?],
        GeoJson::FeatureCollection(fc) => {
            crs = fc.foreign_members.as_ref().and_then(|m| m.get("crs")).and_then(crs_name);
            fc.features.into_iter().enumerate().map(|(i, f)| geojson_feature(path, i, f)).collect::<Result<_>>()?
        }
    };
    Ok(Layer { features, crs })
}

/// The name of a legacy (2008 specification) `crs` member.
fn crs_name(v: &JsonValue) -> Option<String> {
    v.get("properties").and_then(|p| p.get("name")).and_then(|n| n.as_str()).map(str::to_string)
}

fn geojson_feature(path: &Path, i: usize, f: geojson::Feature) -> Result<Feature> {
    let geometry = match &f.geometry {
        Some(g) => geometry_polygons(path, i, g)?,
        None => MultiPolygon::default(),
    };
    let properties = f.properties.map(|m| m.into_iter().map(|(k, v)| (k, json_value(v))).collect()).unwrap_or_default();
    Ok(Feature { geometry, properties })
}

fn json_value(v: JsonValue) -> Property {
    match v {
        JsonValue::Null => Property::Null,
        JsonValue::Bool(b) => Property::Bool(b),
        JsonValue::Number(n) => n.as_i64().map_or_else(|| Property::Float(n.as_f64().unwrap_or(f64::NAN)), Property::Int),
        JsonValue::String(s) => Property::Str(s),
        other => Property::Str(other.to_string()),
    }
}

fn ring_xy(path: &Path, i: usize, ring: &[geojson::Position]) -> Result<Vec<Xy>> {
    ring.iter().map(|p| match p.as_slice() {
        [x, y, ..] => Ok([*x, *y]),
        _ => Err(Error::file(path, format!("feature {i} has a position with fewer than two coordinates"))),
    }).collect()
}

fn polygon(path: &Path, i: usize, rings: &[Vec<geojson::Position>]) -> Result<Option<Polygon>> {
    let Some((ext, holes)) = rings.split_first() else { return Ok(None) };
    Ok(Some(Polygon { exterior: ring_xy(path, i, ext)?, holes: holes.iter().map(|h| ring_xy(path, i, h)).collect::<Result<_>>()? }))
}

fn geometry_polygons(path: &Path, i: usize, g: &Geometry) -> Result<MultiPolygon> {
    let mut parts = Vec::new();
    collect_parts(path, i, g, &mut parts)?;
    Ok(MultiPolygon { parts })
}

fn collect_parts(path: &Path, i: usize, g: &Geometry, out: &mut Vec<Polygon>) -> Result<()> {
    match &g.value {
        GeometryValue::Polygon { coordinates } => out.extend(polygon(path, i, coordinates)?),
        GeometryValue::MultiPolygon { coordinates } => {
            for p in coordinates {
                out.extend(polygon(path, i, p)?);
            }
        }
        GeometryValue::GeometryCollection { geometries } => {
            for g in geometries {
                collect_parts(path, i, g, out)?;
            }
        }
        other => return Err(Error::file(path, format!("feature {i} is a {}, not a polygon", other.type_name()))),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-masks-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn geojson_polygons_holes_multipolygons_properties() {
        let d = tmp("gj");
        let f = d.join("plots.geojson");
        std::fs::write(&f, r#"{"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:28355"}},
            "features": [
              {"type": "Feature", "properties": {"plot": 7, "name": "a", "area": 2.5, "ok": true, "x": null},
               "geometry": {"type": "Polygon", "coordinates": [[[0,0],[4,0],[4,4],[0,4],[0,0]], [[1,1],[2,1],[2,2],[1,2],[1,1]]]}},
              {"type": "Feature", "properties": {"plot": 8},
               "geometry": {"type": "MultiPolygon", "coordinates": [[[[10,0],[11,0],[11,1],[10,0]]], [[[20,0,5],[21,0,5],[21,1,5],[20,0,5]]]]}},
              {"type": "Feature", "properties": {}, "geometry": null}
            ]}"#).unwrap();
        let layer = read_polygons(&f, None).unwrap();
        assert_eq!(layer.crs.as_deref(), Some("EPSG:28355"));
        assert_eq!(layer.features.len(), 3);
        let a = &layer.features[0];
        assert_eq!(a.geometry.parts.len(), 1);
        assert_eq!(a.geometry.parts[0].holes.len(), 1);
        assert!(a.properties.contains(&("plot".into(), Property::Int(7))));
        assert!(a.properties.contains(&("area".into(), Property::Float(2.5))));
        assert!(a.properties.contains(&("ok".into(), Property::Bool(true))));
        assert!(a.properties.contains(&("x".into(), Property::Null)));
        assert_eq!(layer.features[1].geometry.parts.len(), 2);
        assert_eq!(layer.features[1].geometry.parts[1].exterior[0], [20.0, 0.0]);
        assert!(layer.features[2].geometry.parts.is_empty());
        // Directory plus layer name.
        assert_eq!(read_polygons(&d, Some("plots")).unwrap(), layer);
        assert!(read_polygons(&d, Some("nope")).unwrap_err().to_string().contains("plots.geojson"));
    }

    #[test]
    fn geojson_rejects_lines() {
        let d = tmp("gjline");
        let f = d.join("l.geojson");
        std::fs::write(&f, r#"{"type": "LineString", "coordinates": [[0,0],[1,1]]}"#).unwrap();
        let e = read_polygons(&f, None).unwrap_err().to_string();
        assert!(e.contains("feature 0 is a LineString, not a polygon"), "{e}");
    }

    #[test]
    fn shapefile_round_trip_with_holes_and_attributes() {
        use shapefile::dbase::{FieldName, FieldValue, Record, TableWriterBuilder};
        use shapefile::{Point as SP, PolygonRing};
        let d = tmp("shp");
        let f = d.join("stands.shp");
        let table = TableWriterBuilder::new().add_numeric_field(FieldName::try_from("ID").unwrap(), 10, 0).add_character_field(FieldName::try_from("NAME").unwrap(), 20);
        let mut w = shapefile::Writer::from_path(&f, table).unwrap();
        let outer = |x0: f64, s: f64| PolygonRing::Outer(vec![SP::new(x0, 0.0), SP::new(x0, s), SP::new(x0 + s, s), SP::new(x0 + s, 0.0), SP::new(x0, 0.0)]);
        let inner = PolygonRing::Inner(vec![SP::new(1.0, 1.0), SP::new(2.0, 1.0), SP::new(2.0, 2.0), SP::new(1.0, 2.0), SP::new(1.0, 1.0)]);
        // Two exterior rings; the hole is written after the second but lies in the first.
        let poly = shapefile::Polygon::with_rings(vec![outer(0.0, 4.0), outer(10.0, 1.0), inner]);
        let mut rec = Record::default();
        rec.insert("ID".into(), FieldValue::Numeric(Some(3.0)));
        rec.insert("NAME".into(), FieldValue::Character(Some("north".into())));
        w.write_shape_and_record(&poly, &rec).unwrap();
        drop(w);
        std::fs::write(d.join("stands.prj"), "PROJCS[\"test\"]\n").unwrap();
        let layer = read_polygons(&f, None).unwrap();
        assert_eq!(layer.crs.as_deref(), Some("PROJCS[\"test\"]"));
        assert_eq!(layer.features.len(), 1);
        let g = &layer.features[0].geometry;
        assert_eq!(g.parts.len(), 2);
        assert_eq!(g.parts[0].holes.len(), 1);
        assert_eq!(g.parts[1].holes.len(), 0);
        assert_eq!(layer.features[0].properties, vec![("ID".into(), Property::Int(3)), ("NAME".into(), Property::Str("north".into()))]);
    }

    #[test]
    fn missing_and_unsupported_files() {
        assert!(read_polygons("/nonexistent/x.shp", None).is_err());
        let d = tmp("unsup");
        let f = d.join("a.gpkg");
        std::fs::write(&f, "x").unwrap();
        assert!(matches!(read_polygons(&f, None), Err(Error::UnsupportedFormat(_))));
    }
}
