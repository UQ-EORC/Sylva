// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Coordinate reference systems and reprojection.
//!
//! A CRS is given as an EPSG code (`"EPSG:7855"`, `"7855"`, a compound
//! `"EPSG:7855+5711"` or an OGC URN), a PROJ string (`"+proj=utm ..."`) or
//! WKT (WKT1, ESRI WKT1 or WKT2). EPSG definitions come from the
//! `crs-definitions` tables (horizontal CRSs with codes up to 65535) and
//! coordinates are transformed with `proj4rs`, both pure Rust.
//!
//! What is exact and what is not:
//!
//! * A change of projection on the same datum (e.g. MGA zone 55 to GDA2020
//!   geographic, or UTM to WGS 84 longitude and latitude) is exact to float
//!   rounding: only the projection formulas are involved.
//! * A change of datum by a Helmert (`towgs84`) transformation applies the
//!   published parameters exactly, through WGS 84, and changes heights as
//!   well as positions (the heights are treated as ellipsoidal).
//! * A change of datum where one side has no transformation parameters (for
//!   example GDA2020, whose EPSG definition carries none) is not applied:
//!   coordinates are carried across unchanged in latitude and longitude, a
//!   null transformation, reported as approximate.
//! * Datum grids (NTv2, NADCON, geoid models) are not available, so
//!   transformations that need one (NAD27, grid-based vertical datums) are
//!   refused rather than silently approximated, and a change of vertical
//!   datum (e.g. AHD to ellipsoidal heights) is not applied.
//!
//! Geographic coordinates are x = longitude and y = latitude in degrees
//! (the GIS order, not the EPSG axis order).

use std::path::Path;

use proj4rs::transform::{Transform as ProjTransform, TransformClosure};
use proj4rs::Proj;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::Point;

const CHUNK: usize = 1 << 16;

// ---------------------------------------------------------------------- WKT

/// One bracketed WKT node, `KEYWORD[item, item, ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct WktNode {
    pub keyword: String,
    pub items: Vec<WktItem>,
}

/// An item inside a WKT node.
#[derive(Debug, Clone, PartialEq)]
pub enum WktItem {
    Node(WktNode),
    /// A quoted string.
    Text(String),
    /// A number or bare word (`EAST`, `6378137`).
    Word(String),
}

impl WktNode {
    /// The first quoted string (the name of most nodes).
    pub fn name(&self) -> Option<&str> {
        self.items.iter().find_map(|i| if let WktItem::Text(s) = i { Some(s.as_str()) } else { None })
    }

    /// Direct child nodes whose keyword is one of `keywords` (case-insensitive).
    pub fn children(&self, keywords: &[&str]) -> Vec<&WktNode> {
        self.items
            .iter()
            .filter_map(|i| match i {
                WktItem::Node(n) if keywords.iter().any(|k| n.keyword.eq_ignore_ascii_case(k)) => Some(n),
                _ => None,
            })
            .collect()
    }

    pub fn child(&self, keywords: &[&str]) -> Option<&WktNode> {
        self.children(keywords).into_iter().next()
    }

    /// First node with one of `keywords` anywhere below (depth first, self included).
    pub fn find(&self, keywords: &[&str]) -> Option<&WktNode> {
        if keywords.iter().any(|k| self.keyword.eq_ignore_ascii_case(k)) {
            return Some(self);
        }
        self.items.iter().find_map(|i| if let WktItem::Node(n) = i { n.find(keywords) } else { None })
    }

    /// Numeric items in order (quoted strings skipped).
    pub fn numbers(&self) -> Vec<f64> {
        self.items.iter().filter_map(|i| if let WktItem::Word(w) = i { w.parse::<f64>().ok() } else { None }).collect()
    }

    /// The EPSG code of a direct `AUTHORITY["EPSG","n"]` (WKT1) or `ID["EPSG",n]` (WKT2) child.
    pub fn epsg(&self) -> Option<u32> {
        self.children(&["AUTHORITY", "ID"]).into_iter().find_map(|a| {
            let mut it = a.items.iter();
            let auth = match it.next()? {
                WktItem::Text(s) | WktItem::Word(s) => s,
                _ => return None,
            };
            if !auth.eq_ignore_ascii_case("EPSG") {
                return None;
            }
            match it.next()? {
                WktItem::Text(s) | WktItem::Word(s) => s.trim().parse::<u32>().ok(),
                _ => None,
            }
        })
    }
}

struct WktParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl WktParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn err(&self, what: &str) -> Error {
        Error::invalid(format!("malformed WKT at character {}: {what}", self.i))
    }

    /// One WKT node: a keyword, then its items in brackets.
    ///
    /// WKT nests — `PROJCS["...", GEOGCS[...], PROJECTION[...], ...]` — so this
    /// calls itself for each item that is itself a node, and returns when the
    /// closing bracket is reached. Either bracket style is accepted, since both
    /// appear in the wild, and the one that opened a node must close it.
    fn node(&mut self) -> Result<WktNode> {
        self.ws();
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_') {
            self.i += 1;
        }
        if self.i == start {
            return Err(self.err("expected a keyword"));
        }
        let keyword = String::from_utf8_lossy(&self.s[start..self.i]).to_string();
        self.ws();
        let close = match self.s.get(self.i) {
            Some(b'[') => b']',
            Some(b'(') => b')',
            _ => return Err(self.err("expected '[' or '('")),
        };
        self.i += 1;
        let mut items = Vec::new();
        loop {
            self.ws();
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated node")),
                Some(&c) if c == close => {
                    self.i += 1;
                    break;
                }
                Some(b',') => {
                    self.i += 1;
                    continue;
                }
                Some(b'"') => {
                    self.i += 1;
                    let mut text = Vec::new();
                    loop {
                        match self.s.get(self.i) {
                            None => return Err(self.err("unterminated string")),
                            Some(b'"') if self.s.get(self.i + 1) == Some(&b'"') => {
                                text.push(b'"');
                                self.i += 2;
                            }
                            Some(b'"') => {
                                self.i += 1;
                                break;
                            }
                            Some(&c) => {
                                text.push(c);
                                self.i += 1;
                            }
                        }
                    }
                    items.push(WktItem::Text(String::from_utf8_lossy(&text).to_string()));
                }
                Some(_) => {
                    let start = self.i;
                    while self.i < self.s.len() && !matches!(self.s[self.i], b',' | b']' | b')' | b'[' | b'(') {
                        self.i += 1;
                    }
                    if matches!(self.s.get(self.i), Some(b'[') | Some(b'(')) {
                        self.i = start;
                        items.push(WktItem::Node(self.node()?));
                    } else {
                        items.push(WktItem::Word(String::from_utf8_lossy(&self.s[start..self.i]).trim().to_string()));
                    }
                }
            }
        }
        Ok(WktNode { keyword, items })
    }
}

/// Parse WKT (version 1 or 2) into a tree.
pub fn parse_wkt(text: &str) -> Result<WktNode> {
    let text = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    let mut p = WktParser { s: text.as_bytes(), i: 0 };
    let node = p.node()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(node)
}

const HORIZONTAL: &[&str] = &["PROJCS", "GEOGCS", "GEOCCS", "PROJCRS", "PROJECTEDCRS", "GEOGCRS", "GEOGRAPHICCRS", "GEODCRS", "GEODETICCRS"];
const VERTICAL: &[&str] = &["VERT_CS", "VERTCRS", "VERTICALCRS"];

/// The horizontal and vertical components of a (possibly compound or bound) WKT CRS.
fn components(node: &WktNode) -> (Option<&WktNode>, Option<&WktNode>) {
    let kw = node.keyword.to_ascii_uppercase();
    if HORIZONTAL.contains(&kw.as_str()) {
        (Some(node), None)
    } else if VERTICAL.contains(&kw.as_str()) {
        (None, Some(node))
    } else if kw == "COMPD_CS" || kw == "COMPOUNDCRS" {
        (node.child(HORIZONTAL), node.child(VERTICAL))
    } else if kw == "BOUNDCRS" {
        match node.child(&["SOURCECRS"]).and_then(|s| s.items.iter().find_map(|i| if let WktItem::Node(n) = i { Some(n) } else { None })) {
            Some(n) => components(n),
            None => (None, None),
        }
    } else {
        (None, None)
    }
}

fn param(params: &[(String, f64)], names: &[&str]) -> Option<f64> {
    names.iter().find_map(|n| params.iter().find(|(k, _)| k == n).map(|(_, v)| *v))
}

fn fmt_num(v: f64) -> String {
    format!("{v}")
}

/// Datum and ellipsoid part of a PROJ string from a WKT1 `GEOGCS`.
fn wkt1_datum_proj4(geog: &WktNode) -> Result<String> {
    let datum = geog.child(&["DATUM"]).ok_or_else(|| Error::invalid("WKT GEOGCS has no DATUM"))?;
    let sph = datum.child(&["SPHEROID", "ELLIPSOID"]).ok_or_else(|| Error::invalid("WKT DATUM has no SPHEROID"))?;
    let v = sph.numbers();
    if v.len() < 2 || v[0].is_nan() || v[0] <= 0.0 {
        return Err(Error::invalid("WKT SPHEROID needs a semi-major axis and inverse flattening"));
    }
    let mut s = if v[1] == 0.0 { format!("+a={} +b={}", fmt_num(v[0]), fmt_num(v[0])) } else { format!("+a={} +rf={}", fmt_num(v[0]), fmt_num(v[1])) };
    if let Some(t) = datum.child(&["TOWGS84"]) {
        let p = t.numbers();
        if p.len() == 3 || p.len() == 7 {
            s += &format!(" +towgs84={}", p.iter().map(|x| fmt_num(*x)).collect::<Vec<_>>().join(","));
        }
    } else if datum_key(datum.name().unwrap_or("")).map(|k| k == "wgs1984").unwrap_or(false) {
        s += " +towgs84=0,0,0";
    }
    if let Some(pm) = geog.child(&["PRIMEM"]) {
        if let Some(&lon) = pm.numbers().first() {
            if lon != 0.0 {
                s += &format!(" +pm={}", fmt_num(lon));
            }
        }
    }
    Ok(s)
}

/// A PROJ string for a WKT1 `PROJCS` or `GEOGCS` that has no EPSG code
/// (ESRI WKT, custom definitions). Covers the projections forestry data
/// comes in: transverse Mercator, Lambert conformal conic, Albers, Mercator,
/// azimuthal equal area, polar and oblique stereographic, equirectangular.
pub fn wkt1_to_proj4(node: &WktNode) -> Result<String> {
    let kw = node.keyword.to_ascii_uppercase();
    if kw == "GEOGCS" {
        return Ok(format!("+proj=longlat {} +no_defs", wkt1_datum_proj4(node)?));
    }
    if kw != "PROJCS" {
        return Err(Error::invalid(format!("cannot build a PROJ definition from a WKT {} without an EPSG code", node.keyword)));
    }
    let geog = node.child(&["GEOGCS"]).ok_or_else(|| Error::invalid("WKT PROJCS has no GEOGCS"))?;
    let datum = wkt1_datum_proj4(geog)?;
    let method = node.child(&["PROJECTION"]).and_then(|p| p.name()).ok_or_else(|| Error::invalid("WKT PROJCS has no PROJECTION"))?;
    let unit = node.child(&["UNIT"]).and_then(|u| u.numbers().first().copied()).unwrap_or(1.0);
    if unit.is_nan() || unit <= 0.0 {
        return Err(Error::invalid("WKT PROJCS has an invalid linear UNIT"));
    }
    let params: Vec<(String, f64)> = node
        .children(&["PARAMETER"])
        .into_iter()
        .filter_map(|p| Some((p.name()?.to_ascii_lowercase().replace(' ', "_"), *p.numbers().first()?)))
        .collect();
    let lon_0 = param(&params, &["central_meridian", "longitude_of_center", "longitude_of_origin", "longitude_of_natural_origin"]).unwrap_or(0.0);
    let lat_0 = param(&params, &["latitude_of_origin", "latitude_of_center", "latitude_of_natural_origin"]).unwrap_or(0.0);
    let k = param(&params, &["scale_factor", "scale_factor_at_natural_origin"]);
    let lat_1 = param(&params, &["standard_parallel_1", "standard_parallel1"]);
    let lat_2 = param(&params, &["standard_parallel_2", "standard_parallel2"]);
    // False easting and northing are in the CRS's linear unit; PROJ wants metres.
    let x_0 = param(&params, &["false_easting"]).unwrap_or(0.0) * unit;
    let y_0 = param(&params, &["false_northing"]).unwrap_or(0.0) * unit;
    let m = method.to_ascii_lowercase().replace(' ', "_");
    let body = match m.as_str() {
        "transverse_mercator" | "gauss_kruger" => format!("+proj=tmerc +lat_0={} +lon_0={} +k={}", fmt_num(lat_0), fmt_num(lon_0), fmt_num(k.unwrap_or(1.0))),
        "lambert_conformal_conic_2sp" | "lambert_conformal_conic" => {
            let l1 = lat_1.unwrap_or(lat_0);
            let mut s = format!("+proj=lcc +lat_1={} +lat_2={} +lat_0={} +lon_0={}", fmt_num(l1), fmt_num(lat_2.unwrap_or(l1)), fmt_num(lat_0), fmt_num(lon_0));
            if let Some(k) = k {
                s += &format!(" +k_0={}", fmt_num(k));
            }
            s
        }
        "lambert_conformal_conic_1sp" => format!("+proj=lcc +lat_1={} +lat_0={} +lon_0={} +k_0={}", fmt_num(lat_0), fmt_num(lat_0), fmt_num(lon_0), fmt_num(k.unwrap_or(1.0))),
        "mercator_1sp" | "mercator" => match lat_1 {
            Some(ts) => format!("+proj=merc +lat_ts={} +lon_0={}", fmt_num(ts), fmt_num(lon_0)),
            None => format!("+proj=merc +lon_0={} +k={}", fmt_num(lon_0), fmt_num(k.unwrap_or(1.0))),
        },
        "mercator_2sp" => format!("+proj=merc +lat_ts={} +lon_0={}", fmt_num(lat_1.unwrap_or(0.0)), fmt_num(lon_0)),
        "albers_conic_equal_area" | "albers" => format!("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}", fmt_num(lat_1.unwrap_or(0.0)), fmt_num(lat_2.unwrap_or(lat_1.unwrap_or(0.0))), fmt_num(lat_0), fmt_num(lon_0)),
        "lambert_azimuthal_equal_area" => format!("+proj=laea +lat_0={} +lon_0={}", fmt_num(lat_0), fmt_num(lon_0)),
        "polar_stereographic" => {
            let pole = if lat_0 < 0.0 { -90.0 } else { 90.0 };
            let mut s = format!("+proj=stere +lat_0={} +lon_0={}", fmt_num(pole), fmt_num(lon_0));
            if lat_0.abs() != 90.0 {
                s += &format!(" +lat_ts={}", fmt_num(lat_0));
            }
            s += &format!(" +k={}", fmt_num(k.unwrap_or(1.0)));
            s
        }
        "oblique_stereographic" | "double_stereographic" => format!("+proj=sterea +lat_0={} +lon_0={} +k={}", fmt_num(lat_0), fmt_num(lon_0), fmt_num(k.unwrap_or(1.0))),
        "equirectangular" | "plate_carree" | "equidistant_cylindrical" => format!("+proj=eqc +lat_ts={} +lon_0={}", fmt_num(lat_1.unwrap_or(0.0)), fmt_num(lon_0)),
        _ => return Err(Error::invalid(format!("WKT projection {method:?} is not supported without an EPSG code"))),
    };
    let units = if unit == 1.0 { "+units=m".to_string() } else { format!("+to_meter={}", fmt_num(unit)) };
    Ok(format!("{body} +x_0={} +y_0={} {datum} {units} +no_defs", fmt_num(x_0), fmt_num(y_0)))
}

/// A normalised datum name for comparison: lower-case letters and digits,
/// ESRI's `D_` prefix dropped, and the usual spellings of WGS 84 unified.
fn datum_key(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let lower = lower.strip_prefix("d_").unwrap_or(&lower);
    let k: String = lower.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if k.is_empty() {
        return None;
    }
    Some(match k.as_str() {
        "wgs84" | "wgs1984" | "worldgeodeticsystem1984" | "worldgeodeticsystem1984ensemble" => "wgs1984".into(),
        _ => k,
    })
}

// ---------------------------------------------------------------------- CRS

/// Vertical CRSs that LAS files commonly name, for writing compound WKT:
/// `(code, name, datum name, datum code)`.
const VERTICAL_CRS: &[(u32, &str, &str, u32)] = &[
    (5711, "AHD height", "Australian Height Datum", 5111),
    (9458, "AVWS height", "Australian Vertical Working Surface", 1292),
    (5703, "NAVD88 height", "North American Vertical Datum 1988", 5103),
    (5701, "ODN height", "Ordnance Datum Newlyn", 5101),
    (3855, "EGM2008 height", "EGM2008 geoid", 1027),
    (5773, "EGM96 height", "EGM96 geoid", 5171),
];

/// A parsed coordinate reference system.
#[derive(Debug, Clone, PartialEq)]
pub struct Crs {
    /// The definition as given (trimmed).
    pub definition: String,
    /// Human-readable name.
    pub name: String,
    /// PROJ string of the horizontal part, as proj4rs takes it.
    pub proj4: String,
    /// WKT, when known (given, or from the EPSG tables).
    pub wkt: Option<String>,
    /// EPSG code of the horizontal part, when known.
    pub epsg: Option<u32>,
    /// EPSG code of the vertical part of a compound CRS, when known.
    pub vertical_epsg: Option<u32>,
    /// Datum identity (EPSG code or normalised name), when known.
    pub datum: Option<String>,
}

fn datum_identity(horiz: &WktNode) -> Option<String> {
    let d = horiz.find(&["DATUM", "GEODETICDATUM", "TRF", "ENSEMBLE"])?;
    if let Some(code) = d.epsg() {
        // The WGS 84 datum and its ensemble are one datum for our purposes.
        return Some(if code == 6326 { "wgs1984".into() } else { format!("EPSG:{code}") });
    }
    datum_key(d.name()?)
}

fn not_found(code: u32) -> Error {
    Error::invalid(format!("EPSG:{code} is not in the built-in definitions (horizontal CRSs from the EPSG registry); give the CRS as a PROJ string or WKT instead"))
}

impl Crs {
    /// A CRS from an EPSG code of a horizontal (projected, geographic or
    /// geocentric) CRS.
    pub fn from_epsg(code: u32) -> Result<Crs> {
        let def = u16::try_from(code).ok().and_then(crs_definitions::from_code).ok_or_else(|| not_found(code))?;
        let tree = parse_wkt(def.wkt)?;
        let (horiz, vert) = components(&tree);
        let horiz = horiz.ok_or_else(|| Error::invalid(format!("EPSG:{code} is a vertical CRS; give it as the second part of a compound code, e.g. \"EPSG:7855+{code}\"")))?;
        Ok(Crs {
            definition: format!("EPSG:{code}"),
            name: tree.name().unwrap_or("").to_string(),
            proj4: def.proj4.to_string(),
            wkt: Some(def.wkt.to_string()),
            epsg: Some(code),
            vertical_epsg: vert.and_then(WktNode::epsg),
            datum: datum_identity(horiz),
        })
    }

    /// Parse an EPSG code, PROJ string or WKT.
    ///
    /// # Errors
    /// An empty or unrecognised definition, an EPSG code outside the built-in
    /// tables, or WKT without an EPSG code in a projection not covered by
    /// [`wkt1_to_proj4`].
    pub fn parse(text: &str) -> Result<Crs> {
        // Each form is recognised by how it starts, cheapest test first: a
        // PROJ string by its leading '+', then the names and codes that need
        // only a lookup, and WKT last, since it is the only one that has to be
        // parsed. Trailing NUL bytes come from fixed-width fields in LAS
        // headers, where the text is padded rather than terminated.
        let s = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
        if s.is_empty() {
            return Err(Error::invalid("empty CRS definition"));
        }
        if s.starts_with('+') {
            validate(s)?;
            return Ok(Crs { definition: s.to_string(), name: s.to_string(), proj4: s.to_string(), wkt: None, epsg: None, vertical_epsg: None, datum: None });
        }
        if s.eq_ignore_ascii_case("WGS84") || s.eq_ignore_ascii_case("WGS 84") {
            return Crs::from_epsg(4326);
        }
        let lower = s.to_ascii_lowercase();
        let code_part = lower
            .strip_prefix("urn:ogc:def:crs:")
            .map(|r| r.to_string())
            .unwrap_or_else(|| lower.clone());
        let code_part = code_part.strip_prefix("epsg:").map(|r| r.trim_start_matches(':').trim().to_string()).or_else(|| {
            if code_part.chars().all(|c| c.is_ascii_digit() || c == '+') {
                Some(code_part.clone())
            } else {
                None
            }
        });
        if let Some(codes) = code_part {
            let mut parts = codes.split('+').map(|p| p.trim().trim_start_matches("epsg:").parse::<u32>());
            let h = parts.next().and_then(|r| r.ok()).ok_or_else(|| Error::invalid(format!("cannot read an EPSG code from {s:?}")))?;
            let v = match parts.next() {
                None => None,
                Some(Ok(v)) => Some(v),
                Some(Err(_)) => return Err(Error::invalid(format!("cannot read an EPSG code from {s:?}"))),
            };
            if parts.next().is_some() {
                return Err(Error::invalid(format!("cannot read an EPSG code from {s:?}")));
            }
            let mut crs = Crs::from_epsg(h)?;
            if v.is_some() {
                crs.vertical_epsg = v;
                crs.definition = format!("EPSG:{h}+{}", v.unwrap_or(0));
                if let Some(vn) = v.and_then(|v| VERTICAL_CRS.iter().find(|e| e.0 == v)) {
                    crs.name = format!("{} + {}", crs.name, vn.1);
                }
            }
            return Ok(crs);
        }
        if s.as_bytes()[0].is_ascii_alphabetic() && s.contains(['[', '(']) {
            let tree = parse_wkt(s)?;
            let (horiz, vert) = components(&tree);
            let horiz = horiz.ok_or_else(|| Error::invalid("WKT has no horizontal (projected or geographic) CRS"))?;
            let epsg = horiz.epsg();
            let from_table = epsg.and_then(|c| u16::try_from(c).ok()).and_then(crs_definitions::from_code);
            let proj4 = match from_table {
                Some(def) => def.proj4.to_string(),
                None => wkt1_to_proj4(horiz).map_err(|e| match epsg {
                    Some(c) => Error::invalid(format!("{}; {e}", not_found(c))),
                    None => e,
                })?,
            };
            validate(&proj4)?;
            return Ok(Crs {
                definition: s.to_string(),
                name: tree.name().unwrap_or("").to_string(),
                proj4,
                wkt: Some(s.to_string()),
                epsg: from_table.map(|d| d.code as u32),
                vertical_epsg: vert.and_then(WktNode::epsg),
                datum: datum_identity(horiz),
            });
        }
        Err(Error::invalid(format!("unrecognised CRS {s:?}: give an EPSG code (\"EPSG:7855\"), a PROJ string (\"+proj=...\") or WKT")))
    }

    /// A short label: `EPSG:n` (or `EPSG:h+v`) when the code is known, otherwise the name.
    pub fn label(&self) -> String {
        match (self.epsg, self.vertical_epsg) {
            (Some(h), Some(v)) => format!("EPSG:{h}+{v}"),
            (Some(h), None) => format!("EPSG:{h}"),
            _ => self.name.clone(),
        }
    }

    /// WKT to store in a file. A given WKT is kept; an EPSG code gets the
    /// registry's WKT, as a `COMPD_CS` when the vertical CRS is one Sylva
    /// knows by name. `None` for a PROJ string (which has no WKT form here).
    pub fn to_wkt(&self) -> Option<String> {
        if self.definition.starts_with('+') {
            return None;
        }
        let wkt = self.wkt.clone()?;
        if !self.definition.to_ascii_lowercase().contains("epsg") {
            return Some(wkt);
        }
        match self.vertical_epsg.and_then(|v| VERTICAL_CRS.iter().find(|e| e.0 == v)) {
            Some((code, name, dname, dcode)) if !wkt.starts_with("COMPD_CS") => Some(format!(
                "COMPD_CS[\"{} + {name}\",{wkt},VERT_CS[\"{name}\",VERT_DATUM[\"{dname}\",2005,AUTHORITY[\"EPSG\",\"{dcode}\"]],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Gravity-related height\",UP],AUTHORITY[\"EPSG\",\"{code}\"]]]",
                self.name.split(" + ").next().unwrap_or(&self.name)
            )),
            _ => Some(wkt),
        }
    }

    fn proj(&self) -> Result<Proj> {
        Proj::from_proj_string(&self.proj4).map_err(|e| proj_error(&self.proj4, e))
    }

    /// Whether coordinates are longitude and latitude (degrees).
    pub fn is_geographic(&self) -> Result<bool> {
        Ok(self.proj()?.is_latlong())
    }
}

/// Check a PROJ string; one that only fails for want of a datum grid is
/// accepted here (it can label data) and refused when transforming.
fn validate(proj4: &str) -> Result<()> {
    match Proj::from_proj_string(proj4) {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().to_ascii_lowercase().contains("grid") => Ok(()),
        Err(e) => Err(proj_error(proj4, e)),
    }
}

fn proj_error(def: &str, e: proj4rs::errors::Error) -> Error {
    let msg = e.to_string();
    if msg.to_ascii_lowercase().contains("grid") {
        Error::invalid(format!("{def:?} needs a datum grid ({msg}); datum grids are not available, so give the CRS with +towgs84 Helmert parameters instead"))
    } else {
        Error::invalid(format!("invalid CRS {def:?}: {msg}"))
    }
}

// ------------------------------------------------------------ reprojection

/// How a transformation between two CRSs is carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Same definition: coordinates are unchanged.
    Identity,
    /// Same datum, different projection: exact.
    Conversion,
    /// Datum change by published Helmert (towgs84) parameters: exact to those parameters.
    Helmert,
    /// Different datums without parameters between them: not applied (approximate).
    NullDatum,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Identity => "identity",
            Kind::Conversion => "conversion",
            Kind::Helmert => "helmert",
            Kind::NullDatum => "null datum",
        }
    }
}

/// The transformation [`reproject`] will apply.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub kind: Kind,
    /// Whether the result is exact (to float rounding and any published parameters).
    pub exact: bool,
    /// Whether heights change.
    pub changes_z: bool,
    /// Explanation, naming what is not applied when inexact.
    pub note: String,
}

/// The datum-defining tokens of a PROJ string (ellipsoid, datum, towgs84,
/// nadgrids, prime meridian), as a geographic PROJ string.
fn geographic_of(proj4: &str) -> String {
    const KEEP: &[&str] = &["ellps", "a", "b", "rf", "f", "es", "e", "R", "datum", "towgs84", "nadgrids", "pm"];
    let mut s = String::from("+proj=longlat");
    for tok in proj4.split_whitespace() {
        let t = tok.trim_start_matches('+');
        let key = t.split('=').next().unwrap_or("");
        if KEEP.contains(&key) {
            s.push(' ');
            s.push('+');
            s.push_str(t);
        }
    }
    s.push_str(" +no_defs");
    s
}

/// The datum part of a PROJ string, normalised for comparison.
fn datum_tokens(proj4: &str) -> Vec<String> {
    let mut v: Vec<String> = geographic_of(proj4)
        .split_whitespace()
        .filter(|t| !t.starts_with("+proj") && !t.starts_with("+no_defs"))
        .map(|t| {
            // Zero Helmert parameters are all the same parameters.
            if let Some(p) = t.strip_prefix("+towgs84=") {
                if p.split(',').all(|x| x.trim().parse::<f64>().map(|x| x == 0.0).unwrap_or(false)) {
                    return "+towgs84=0".to_string();
                }
            }
            t.to_string()
        })
        .collect();
    v.sort();
    v
}

fn has_datum_params(proj4: &str) -> bool {
    datum_tokens(proj4).iter().any(|t| t.starts_with("+towgs84=") || (t.starts_with("+datum=") && t != "+datum=none"))
}

/// A PROJ string with its datum shift removed: `+towgs84` dropped and the
/// zero-parameter datums replaced by their ellipsoids.
fn without_datum_shift(proj4: &str) -> String {
    proj4
        .split_whitespace()
        .filter(|t| !t.starts_with("+towgs84="))
        .map(|t| match t.to_ascii_lowercase().as_str() {
            "+datum=wgs84" => "+ellps=WGS84",
            "+datum=nad83" => "+ellps=GRS80",
            _ => t,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn zero_helmert(proj4: &str) -> bool {
    datum_tokens(proj4).iter().any(|t| t == "+towgs84=0" || t.eq_ignore_ascii_case("+datum=WGS84") || t.eq_ignore_ascii_case("+datum=NAD83"))
}

fn same_datum(src: &Crs, dst: &Crs) -> Option<bool> {
    match (&src.datum, &dst.datum) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    }
}

/// Work out how `src` coordinates become `dst` coordinates.
pub fn plan(src: &Crs, dst: &Crs) -> Result<Plan> {
    src.proj()?;
    dst.proj()?;
    let vertical_note = match (src.vertical_epsg, dst.vertical_epsg) {
        (Some(a), Some(b)) if a != b => Some(format!("the change of vertical datum from EPSG:{a} to EPSG:{b} needs a geoid grid and is not applied")),
        _ => None,
    };
    let finish = |kind: Kind, exact: bool, changes_z: bool, note: String| {
        let (exact, note) = match &vertical_note {
            Some(v) => (false, if note.is_empty() { v.clone() } else { format!("{note}; {v}") }),
            None => (exact, note),
        };
        Ok(Plan { kind, exact, changes_z, note })
    };
    if src.proj4 == dst.proj4 {
        return finish(Kind::Identity, true, false, "same coordinate system".into());
    }
    // Probe whether proj4rs applies a datum shift between the two datums.
    let (gs, gd) = (geographic_of(&src.proj4), geographic_of(&dst.proj4));
    let gps = Proj::from_proj_string(&gs).map_err(|e| proj_error(&gs, e))?;
    let gpd = Proj::from_proj_string(&gd).map_err(|e| proj_error(&gd, e))?;
    let mut probe = (0.4_f64, -0.6_f64, 0.0_f64);
    proj4rs::transform::transform(&gps, &gpd, &mut probe).map_err(|e| Error::invalid(format!("datum transformation failed: {e}")))?;
    let shifted = probe != (0.4, -0.6, 0.0);
    if shifted && !(zero_helmert(&src.proj4) && zero_helmert(&dst.proj4)) {
        return finish(Kind::Helmert, true, true, "datum change by the published Helmert (towgs84) parameters, through WGS 84; heights are treated as ellipsoidal".into());
    }
    // Datums tied to WGS 84 by zero parameters (WGS 84, GDA94, NAD83,
    // ETRS89, NZGD2000 as the EPSG definitions give them) are one datum here.
    let same = datum_tokens(&src.proj4) == datum_tokens(&dst.proj4);
    let conversion = (zero_helmert(&src.proj4) && zero_helmert(&dst.proj4))
        || match same_datum(src, dst) {
            Some(eq) => eq || (same && has_datum_params(&src.proj4)),
            None => same,
        };
    if conversion {
        finish(Kind::Conversion, true, false, "change of projection on the same datum".into())
    } else {
        let names = |c: &Crs| c.datum.clone().unwrap_or_else(|| c.label());
        finish(
            Kind::NullDatum,
            false,
            false,
            format!("no transformation parameters are known between the datums of {} and {} ({} and {}); latitude and longitude are carried across unchanged (a null transformation)", src.label(), dst.label(), names(src), names(dst)),
        )
    }
}

/// proj4rs adaptor over a slice of points: a point that fails becomes NaN
/// rather than stopping the whole transformation.
struct Points<'a>(&'a mut [Point]);

impl ProjTransform for Points<'_> {
    fn transform_coordinates<F: TransformClosure>(&mut self, f: &mut F) -> proj4rs::errors::Result<()> {
        for p in self.0.iter_mut() {
            if p[0].is_nan() {
                continue;
            }
            *p = match f(p[0], p[1], p[2]) {
                Ok((x, y, z)) => [x, y, z],
                Err(_) => [f64::NAN; 3],
            };
        }
        Ok(())
    }
}

/// Transform points from `src` to `dst` in place, in parallel. Points that
/// cannot be transformed (outside a projection's domain, NaN input) become
/// NaN. Geographic coordinates are x = longitude, y = latitude in degrees.
pub fn reproject(xyz: &mut [Point], src: &Crs, dst: &Crs) -> Result<Plan> {
    let plan = plan(src, dst)?;
    if plan.kind == Kind::Identity {
        return Ok(plan);
    }
    let (ps, pd) = if plan.kind == Kind::Conversion {
        // Same datum: leave out any datum shift, so latitude and longitude
        // carry across unchanged (proj4rs would otherwise pass WGS 84 and
        // GRS80 datums tied by zero parameters through geocentric space).
        let (s, d) = (without_datum_shift(&src.proj4), without_datum_shift(&dst.proj4));
        (Proj::from_proj_string(&s).map_err(|e| proj_error(&s, e))?, Proj::from_proj_string(&d).map_err(|e| proj_error(&d, e))?)
    } else {
        (src.proj()?, dst.proj()?)
    };
    let (src_deg, dst_deg) = (ps.is_latlong(), pd.is_latlong());
    xyz.par_chunks_mut(CHUNK).for_each(|chunk| {
        for p in chunk.iter_mut() {
            if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
                *p = [f64::NAN; 3];
            } else if src_deg {
                p[0] = p[0].to_radians();
                p[1] = p[1].to_radians();
            }
        }
        let mut pts = Points(chunk);
        let _ = proj4rs::transform::transform(&ps, &pd, &mut pts);
        if dst_deg {
            for p in chunk.iter_mut() {
                p[0] = p[0].to_degrees();
                p[1] = p[1].to_degrees();
            }
        }
    });
    Ok(plan)
}

// --------------------------------------------------------------------- LAS

/// The CRS a LAS/LAZ header declares: the WKT of an OGC WKT VLR/EVLR if
/// present, otherwise `EPSG:n` (or `EPSG:h+v`) from the GeoTIFF keys.
/// `None` when neither is present or the keys are user-defined.
pub fn las_header_crs(header: &las::Header) -> Option<String> {
    if let Some(bytes) = header.get_wkt_crs_bytes() {
        let text = String::from_utf8_lossy(bytes);
        let text = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }
    let keys = header.get_geotiff_crs().ok().flatten()?;
    let valid = |k: u16| (1024..=32766).contains(&k).then_some(k);
    let h = keys.get_projected_crs_geo_key_value().and_then(valid).or_else(|| keys.get_geodetic_crs_geo_key_value().and_then(valid))?;
    Some(match keys.get_vertical_crs_geo_key_value().and_then(valid) {
        Some(v) => format!("EPSG:{h}+{v}"),
        None => format!("EPSG:{h}"),
    })
}

/// Read only the header of a LAS/LAZ file and return its CRS (see [`las_header_crs`]).
pub fn read_las_crs(path: impl AsRef<Path>) -> Result<Option<String>> {
    let reader = las::Reader::from_path(path.as_ref())?;
    Ok(las_header_crs(reader.header()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn one(x: f64, y: f64, z: f64, src: &str, dst: &str) -> (Point, Plan) {
        let mut p = vec![[x, y, z]];
        let plan = reproject(&mut p, &Crs::parse(src).unwrap(), &Crs::parse(dst).unwrap()).unwrap();
        (p[0], plan)
    }

    #[test]
    fn parses_codes() {
        for s in ["EPSG:7855", "epsg:7855", "7855", " EPSG::7855 ", "urn:ogc:def:crs:EPSG::7855"] {
            let c = Crs::parse(s).unwrap();
            assert_eq!(c.epsg, Some(7855), "{s}");
            assert_eq!(c.label(), "EPSG:7855");
            assert_eq!(c.name, "GDA2020 / MGA zone 55");
        }
        let c = Crs::parse("EPSG:7855+5711").unwrap();
        assert_eq!((c.epsg, c.vertical_epsg), (Some(7855), Some(5711)));
        assert_eq!(c.label(), "EPSG:7855+5711");
        let w = c.to_wkt().unwrap();
        assert!(w.starts_with("COMPD_CS[\"GDA2020 / MGA zone 55 + AHD height\""));
        let back = Crs::parse(&w).unwrap();
        assert_eq!((back.epsg, back.vertical_epsg), (Some(7855), Some(5711)));
        assert!(Crs::parse("EPSG:99999").is_err());
        assert!(Crs::parse("EPSG:abc").is_err());
        assert!(Crs::parse("").is_err());
        assert!(Crs::parse("not a crs").is_err());
        assert!(Crs::parse("+proj=nonsense").is_err());
        assert_eq!(Crs::parse("WGS84").unwrap().epsg, Some(4326));
    }

    #[test]
    fn wkt_parser_handles_nesting_and_quotes() {
        let t = parse_wkt(r#"PROJCRS["a ""b""",BASEGEOGCRS["x",DATUM["d",ELLIPSOID["e",6378137,298.257]]],ID["EPSG",32755]]"#).unwrap();
        assert_eq!(t.name(), Some("a \"b\""));
        assert_eq!(t.epsg(), Some(32755));
        assert!(parse_wkt("PROJCS[\"x\"").is_err());
        assert!(parse_wkt("PROJCS[\"x\"] junk").is_err());
    }

    #[test]
    fn wkt2_with_id_uses_the_table() {
        let c = Crs::parse(r#"PROJCRS["WGS 84 / UTM zone 55S",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]]],CONVERSION["UTM zone 55S",METHOD["Transverse Mercator"]],ID["EPSG",32755]]"#).unwrap();
        assert_eq!(c.epsg, Some(32755));
        assert_eq!(c.datum.as_deref(), Some("wgs1984"));
    }

    fn strip_authorities(n: &WktNode) -> WktNode {
        WktNode {
            keyword: n.keyword.clone(),
            items: n
                .items
                .iter()
                .filter(|i| !matches!(i, WktItem::Node(c) if c.keyword == "AUTHORITY"))
                .map(|i| if let WktItem::Node(c) = i { WktItem::Node(strip_authorities(c)) } else { i.clone() })
                .collect(),
        }
    }

    fn to_text(n: &WktNode) -> String {
        let items: Vec<String> = n
            .items
            .iter()
            .map(|i| match i {
                WktItem::Node(c) => to_text(c),
                WktItem::Text(s) => format!("\"{s}\""),
                WktItem::Word(w) => w.clone(),
            })
            .collect();
        format!("{}[{}]", n.keyword, items.join(","))
    }

    #[test]
    fn wkt1_without_codes_matches_the_epsg_definition() {
        // Strip every AUTHORITY so the PROJ string is built from the WKT itself.
        for (code, x, y) in [(7855u32, 512_345.678, 5_412_345.678), (32755, 512_345.678, 5_412_345.678), (3577, 1_234_567.0, -3_456_789.0), (7845, 1_234_567.0, -3_456_789.0), (3031, 100_000.0, 1_500_000.0), (2193, 1_570_000.0, 5_180_000.0)] {
            let def = crs_definitions::from_code(code as u16).unwrap();
            let bare = to_text(&strip_authorities(&parse_wkt(def.wkt).unwrap()));
            let c = Crs::parse(&bare).unwrap();
            assert_eq!(c.epsg, None);
            let (a, _) = one(x, y, 0.0, &format!("EPSG:{code}"), "EPSG:4326");
            let mut p = vec![[x, y, 0.0]];
            reproject(&mut p, &c, &Crs::from_epsg(4326).unwrap()).unwrap();
            assert!(close(a[0], p[0][0], 1e-9) && close(a[1], p[0][1], 1e-9), "EPSG:{code}: {a:?} vs {:?} ({})", p[0], c.proj4);
        }
    }

    #[test]
    fn esri_wkt_in_feet() {
        // A US State Plane zone in US survey feet: false easting given in feet.
        let wkt = r#"PROJCS["NAD_1983_StatePlane_Texas_Central_FIPS_4203_Feet",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",2296583.333333333],PARAMETER["False_Northing",9842500.0],PARAMETER["Central_Meridian",-100.3333333333333],PARAMETER["Standard_Parallel_1",30.11666666666667],PARAMETER["Standard_Parallel_2",31.88333333333333],PARAMETER["Latitude_Of_Origin",29.66666666666667],UNIT["Foot_US",0.3048006096012192]]"#;
        let c = Crs::parse(wkt).unwrap();
        let mut p = vec![[-100.3333333333333, 29.66666666666667, 0.0]];
        reproject(&mut p, &Crs::parse("+proj=longlat +ellps=GRS80 +no_defs").unwrap(), &c).unwrap();
        assert!(close(p[0][0], 2_296_583.333333333, 1e-6) && close(p[0][1], 9_842_500.0, 1e-6), "{:?}", p[0]);
    }

    #[test]
    fn projection_round_trip_and_kinds() {
        let (g, plan) = one(512_345.678, 5_412_345.678, 123.4, "EPSG:7855", "EPSG:7844");
        assert_eq!(plan.kind, Kind::Conversion);
        assert!(plan.exact && !plan.changes_z);
        assert_eq!(g[2], 123.4);
        let (back, _) = one(g[0], g[1], g[2], "EPSG:7844", "EPSG:7855");
        assert!(close(back[0], 512_345.678, 1e-6) && close(back[1], 5_412_345.678, 1e-6));
        let (_, plan) = one(512_345.678, 5_412_345.678, 0.0, "EPSG:7855", "EPSG:7855");
        assert_eq!(plan.kind, Kind::Identity);
        // GDA2020 has no towgs84: to WGS 84 is a null transformation.
        let (_, plan) = one(512_345.678, 5_412_345.678, 0.0, "EPSG:7855", "EPSG:32755");
        assert_eq!(plan.kind, Kind::NullDatum);
        assert!(!plan.exact);
        // GDA94 is towgs84=0: to WGS 84 is a (zero) Helmert, i.e. the same
        // datum, and latitude and longitude carry across unchanged.
        let (a, plan) = one(512_345.678, 5_412_345.678, 7.0, "EPSG:28355", "EPSG:4326");
        assert_eq!(plan.kind, Kind::Conversion);
        let (b, _) = one(512_345.678, 5_412_345.678, 7.0, "EPSG:28355", "EPSG:4283");
        assert_eq!(a, b);
        // OSGB36 carries a 7-parameter Helmert.
        let (_, plan) = one(400_000.0, 300_000.0, 50.0, "EPSG:27700", "EPSG:4326");
        assert_eq!(plan.kind, Kind::Helmert);
        assert!(plan.exact && plan.changes_z);
        // Vertical datum changes are not applied.
        let (_, plan) = one(512_345.678, 5_412_345.678, 0.0, "EPSG:7855+5711", "EPSG:7855+9458");
        assert!(!plan.exact);
    }

    #[test]
    fn grids_are_refused() {
        // NAD27 can label data but cannot be transformed without its grids.
        let nad27 = Crs::parse("EPSG:4267").unwrap();
        let e = plan(&nad27, &Crs::parse("EPSG:4326").unwrap()).unwrap_err().to_string();
        assert!(e.contains("grid"), "{e}");
        assert!(Crs::parse("+proj=longlat +datum=NAD27").is_ok());
    }

    #[test]
    fn nan_and_out_of_domain_become_nan() {
        let mut p = vec![[f64::NAN, 1.0, 2.0], [512_345.0, 5_412_345.0, 1.0], [f64::INFINITY, 0.0, 0.0]];
        reproject(&mut p, &Crs::parse("EPSG:7855").unwrap(), &Crs::parse("EPSG:7844").unwrap()).unwrap();
        assert!(p[0].iter().all(|v| v.is_nan()) && p[2].iter().all(|v| v.is_nan()));
        assert!(p[1][0].is_finite());
    }

    #[test]
    fn parallel_result_matches_serial() {
        let pts: Vec<Point> = (0..200_000).map(|i| [500_000.0 + (i % 1000) as f64, 5_400_000.0 + (i / 1000) as f64, 10.0]).collect();
        let (s, d) = (Crs::parse("EPSG:27700").unwrap(), Crs::parse("EPSG:4326").unwrap());
        let mut par = pts.clone();
        reproject(&mut par, &s, &d).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let mut ser = pts.clone();
        pool.install(|| reproject(&mut ser, &s, &d).unwrap());
        assert_eq!(par, ser);
    }

    #[test]
    fn every_wkt1_projection_method_matches_its_epsg_definition() {
        // Geographic points projected by the EPSG definition and by a PROJ string built
        // from the same WKT with its AUTHORITY codes removed: Mercator 1SP and 2SP,
        // azimuthal equal area, equidistant cylindrical, oblique stereographic (with a
        // 7-parameter Helmert), polar stereographic by pole and by standard parallel,
        // Lambert conformal conic 2SP, transverse Mercator and plain geographic.
        for (code, lon, lat) in [(3395u32, 20.0, 45.0), (3994, 150.0, -30.0), (3035, 10.0, 52.0), (4087, 30.0, 10.0), (28992, 5.4, 52.1), (3413, -45.0, 75.0), (3032, 70.0, -70.0), (5041, 0.0, 85.0), (3976, 30.0, -75.0), (2154, 3.0, 46.5), (3112, 135.0, -25.0), (3006, 15.0, 60.0), (4326, 10.0, 20.0), (4283, 150.0, -30.0)] {
            let def = crs_definitions::from_code(code as u16).unwrap();
            let bare = Crs::parse(&to_text(&strip_authorities(&parse_wkt(def.wkt).unwrap()))).unwrap();
            assert_eq!(bare.epsg, None);
            let (a, _) = one(lon, lat, 0.0, "EPSG:4326", &format!("EPSG:{code}"));
            let mut p = vec![[lon, lat, 0.0]];
            reproject(&mut p, &Crs::from_epsg(4326).unwrap(), &bare).unwrap();
            let tol = if bare.proj4.contains("longlat") { 1e-12 } else { 1e-6 };
            assert!(close(a[0], p[0][0], tol) && close(a[1], p[0][1], tol) && close(a[2], p[0][2], 1e-6), "EPSG:{code}: {a:?} vs {:?} ({})", p[0], bare.proj4);
        }
    }

    #[test]
    fn wkt1_without_enough_to_build_a_definition() {
        let err = |wkt: &str| wkt1_to_proj4(&parse_wkt(wkt).unwrap()).unwrap_err().to_string();
        let geog = r#"GEOGCS["g",DATUM["d",SPHEROID["s",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#;
        assert_eq!(err(r#"VERT_CS["h",VERT_DATUM["d",2005]]"#), "cannot build a PROJ definition from a WKT VERT_CS without an EPSG code");
        assert_eq!(err(&format!(r#"PROJCS["p",{geog},PROJECTION["Hotine_Oblique_Mercator"],UNIT["metre",1]]"#)), "WKT projection \"Hotine_Oblique_Mercator\" is not supported without an EPSG code");
        assert_eq!(err(&format!(r#"PROJCS["p",{geog},PROJECTION["Transverse_Mercator"],UNIT["metre",0]]"#)), "WKT PROJCS has an invalid linear UNIT");
        assert_eq!(err(r#"PROJCS["p",PROJECTION["Transverse_Mercator"]]"#), "WKT PROJCS has no GEOGCS");
        assert_eq!(err(r#"GEOGCS["g",DATUM["d",SPHEROID["s",0,298.257223563]]]"#), "WKT SPHEROID needs a semi-major axis and inverse flattening");
        assert_eq!(err(r#"GEOGCS["g",DATUM["d"]]"#), "WKT DATUM has no SPHEROID");
        // A sphere (inverse flattening 0) and a prime meridian off Greenwich.
        let sphere = wkt1_to_proj4(&parse_wkt(r#"GEOGCS["g",DATUM["d",SPHEROID["s",6371000,0]],PRIMEM["p",2.5]]"#).unwrap()).unwrap();
        assert_eq!(sphere, "+proj=longlat +a=6371000 +b=6371000 +pm=2.5 +no_defs");
    }
}
