// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! RiSCAN PRO projects and RiSCAN PRO's filters.
//!
//! A `.RiSCAN` directory holds `project.rsp` (XML with the POP and every
//! scan position's SOP), `SCANS/ScanPosNNN/SINGLESCANS/*.rxp` and often a
//! `DAT/ScanPosNNN.DAT` copy of each SOP matrix. Older exports use
//! `all_sop.csv` / `project.pop` instead, as do the scanner's own `.PROJ`
//! projects (`ScanPosNNN.SCNPOS/scans/*.rxp`), which also record each
//! position's attitude (`.pose` roll, pitch and yaw, or
//! `pose_estimation.sop`), a GNSS fix (`final.pose`) and the reflective
//! targets the scanner found (`.tpl`). [`read_project`] reads every layout.
//!
//! The readers follow the Python package's original parsers value for value
//! (Python's `float()`, `json`, `csv` and ElementTree semantics, and
//! NumPy's float32 arithmetic in [`angular_steps`] and [`riscan_like_mask`]),
//! so both languages see the same project.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::{filters, Point, Transform};

/// A 4x4 matrix, row-major.
pub type Matrix4 = [f64; 16];

/// Angular scan pattern of a scan (`project.rsp`): zenith `theta` (degrees
/// from up) and azimuth `phi` start, increment and count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanPattern {
    pub theta_start: f64,
    pub theta_delta: f64,
    pub theta_count: i64,
    pub phi_start: f64,
    pub phi_delta: f64,
    pub phi_count: i64,
}

/// One scan position of a project.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScanPosition {
    /// Position name, e.g. `ScanPos001`.
    pub name: String,
    /// Every `.rxp` of the position, monitoring and residual files excluded;
    /// the first is the position's scan.
    pub scans: Vec<PathBuf>,
    /// Scanner -> project transform.
    pub sop: Option<Matrix4>,
    /// Scanner model (`project.rsp` only).
    pub instrument: Option<String>,
    /// Scan pattern of the first scan that records one (`project.rsp` only).
    pub pattern: Option<ScanPattern>,
    /// RIEGL `.tpl` target list (scanner projects).
    pub tiepoints: Option<PathBuf>,
    /// `(latitude, longitude, altitude)` the scanner recorded.
    pub gnss: Option<[f64; 3]>,
    /// Row-major 3x3 rotation from the scanner's frame to a level,
    /// north-referenced one.
    pub attitude: Option<[f64; 9]>,
}

impl ScanPosition {
    /// The position's scan: its first `.rxp`.
    pub fn rxp(&self) -> Option<&Path> {
        self.scans.first().map(|p| p.as_path())
    }
}

/// A parsed project.
#[derive(Debug, Clone, PartialEq)]
pub struct RiscanProject {
    pub path: PathBuf,
    pub positions: Vec<ScanPosition>,
    /// Project -> global matrix (often geocentric).
    pub pop: Option<Matrix4>,
    pub name: String,
}

impl RiscanProject {
    /// Indices of the positions that have a scan (and a SOP, if `require_sop`).
    pub fn with_scans(&self, require_sop: bool) -> Vec<usize> {
        (0..self.positions.len()).filter(|&i| {
            let p = &self.positions[i];
            p.rxp().is_some() && (p.sop.is_some() || !require_sop)
        }).collect()
    }
}

// ------------------------------------------------------- Python's parsing rules

/// Python's `str.isspace()` for the characters that occur in text files.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.split()` without arguments.
fn py_split(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_py_space).filter(|t| !t.is_empty())
}

/// Python's `str.splitlines()`.
fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut chars) = (0, s.char_indices().peekable());
    while let Some((i, c)) = chars.next() {
        if matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = chars.peek() {
                    chars.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Python's `repr()` of a string, for error messages.
fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// Drop the underscores Python allows between digits; None if one is misplaced.
fn without_underscores(t: &str) -> Option<String> {
    let b = t.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
            return None;
        }
    }
    Some(t.replace('_', ""))
}

/// Python's `float(s)` of a string.
pub(crate) fn py_float(s: &str) -> Option<f64> {
    let t = s.trim_matches(is_py_space);
    let t = if t.contains('_') { without_underscores(t)? } else { t.to_string() };
    // Rust also reads "infinity" and "nan" in any case, with a sign, as Python does.
    if t.is_empty() || !t.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'+' | b'-')) {
        return None;
    }
    t.parse::<f64>().ok()
}

fn float_or_err(s: &str) -> Result<f64> {
    py_float(s).ok_or_else(|| Error::invalid(format!("could not convert string to float: {}", py_repr(s))))
}

/// Python's `int(s)` of a string (base 10).
pub(crate) fn py_int(s: &str) -> Option<i64> {
    let t = s.trim_matches(is_py_space);
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit() || c == b'_') {
        return None;
    }
    let t = if t.contains('_') { without_underscores(t)? } else { t.to_string() };
    t.parse::<i64>().ok()
}

fn int_or_err(s: &str) -> Result<i64> {
    py_int(s).ok_or_else(|| Error::invalid(format!("invalid literal for int() with base 10: {}", py_repr(s))))
}

/// The last component of a path without its last suffix (Python's `Path.stem`).
fn py_stem(path: &Path) -> String {
    use std::path::Component;
    let name = match path.components().next_back() {
        Some(Component::Normal(s)) => s.to_string_lossy().into_owned(),
        Some(Component::ParentDir) => "..".into(),
        _ => String::new(),
    };
    if let Some(i) = name.rfind('.') {
        let stem = &name[..i];
        if !stem.trim_start_matches('.').is_empty() {
            return stem.to_string();
        }
    }
    name
}

/// Read a text file as UTF-8, as Python's `read_text()` does here.
fn read_utf8(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    String::from_utf8(bytes).map_err(|e| Error::invalid(format!("{}: not UTF-8: {e}", path.display())))
}

// ------------------------------------------------------------------------ JSON

/// A JSON value as Python's `json.loads` reads it (which also accepts `NaN`,
/// `Infinity` and `-Infinity`; the last of repeated keys wins).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// A number, and whether it was written as an integer.
    Num(f64, bool),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub(crate) fn parse(s: &str) -> Option<Json> {
        let mut p = JsonParser { s: s.as_bytes(), text: s, i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        (p.i == p.s.len()).then_some(v)
    }

    /// `d.get(key)` of a dict (None for anything else).
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python's truth value.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Num(x, _) => *x != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Arr(a) => !a.is_empty(),
            Json::Obj(o) => !o.is_empty(),
        }
    }

    /// Python's `float(v)`; None where it raises.
    pub(crate) fn float(&self) -> Option<f64> {
        match self {
            Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Json::Num(x, _) => Some(*x),
            Json::Str(s) => py_float(s),
            _ => None,
        }
    }

    /// Python's `int(v)`; None where it raises.
    pub(crate) fn int(&self) -> Option<i64> {
        match self {
            Json::Bool(b) => Some(*b as i64),
            Json::Num(x, _) if x.is_finite() => Some(x.trunc() as i64),
            Json::Str(s) => py_int(s),
            _ => None,
        }
    }

    /// Python's `str(v)`.
    pub(crate) fn py_str(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(b) => if *b { "True".into() } else { "False".into() },
            Json::Num(x, true) => format!("{x:.0}"),
            Json::Num(x, false) if x.is_nan() => "nan".into(),
            Json::Num(x, false) if x.is_infinite() => if *x > 0.0 { "inf".into() } else { "-inf".into() },
            Json::Num(x, false) => format!("{x:?}"),
            Json::Str(s) => s.clone(),
            Json::Arr(_) | Json::Obj(_) => format!("{self:?}"),
        }
    }
}

struct JsonParser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Json> {
        match *self.s.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Some(Json::Obj(items));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return None;
                    }
                    let k = self.string()?;
                    self.ws();
                    if !self.eat(":") {
                        return None;
                    }
                    self.ws();
                    let v = self.value()?;
                    items.push((k, v));
                    self.ws();
                    if self.eat("}") {
                        return Some(Json::Obj(items));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Some(Json::Arr(items));
                }
                loop {
                    self.ws();
                    items.push(self.value()?);
                    self.ws();
                    if self.eat("]") {
                        return Some(Json::Arr(items));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            b'n' if self.eat("null") => Some(Json::Null),
            b't' if self.eat("true") => Some(Json::Bool(true)),
            b'f' if self.eat("false") => Some(Json::Bool(false)),
            b'N' if self.eat("NaN") => Some(Json::Num(f64::NAN, false)),
            b'I' if self.eat("Infinity") => Some(Json::Num(f64::INFINITY, false)),
            b'-' if self.eat("-Infinity") => Some(Json::Num(f64::NEG_INFINITY, false)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    /// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return None,
        }
        let mut integer = true;
        if self.s.get(self.i) == Some(&b'.') && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
            digits(self);
            integer = false;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                self.i = save;
            } else {
                integer = false;
            }
        }
        self.text[start..self.i].parse::<f64>().ok().map(|x| Json::Num(x, integer))
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.text.get(self.i..self.i + 4)?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let rest = &self.text[self.i..];
            let c = rest.chars().next()?;
            self.i += c.len_utf8();
            match c {
                '"' => return Some(out),
                '\\' => {
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4() {
                                    Some(l) if (0xdc00..0xe000).contains(&l) => u = 0x10000 + ((u - 0xd800) << 10) + (l - 0xdc00),
                                    _ => self.i = save,
                                }
                            }
                            // A lone surrogate, which Python keeps, has no Rust char.
                            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => out.push(c),
            }
        }
    }
}

/// `np.asarray(v, dtype=float)` for a JSON value: its shape and values.
/// Errors where NumPy raises (ragged nesting, unconvertible entries).
fn json_array(v: &Json) -> Result<(Vec<usize>, Vec<f64>)> {
    fn shape(v: &Json) -> Result<Vec<usize>> {
        match v {
            Json::Arr(items) => {
                let mut inner: Option<Vec<usize>> = None;
                for x in items {
                    let s = shape(x)?;
                    if inner.as_ref().is_some_and(|i| *i != s) {
                        return Err(Error::invalid("setting an array element with a sequence. The requested array has an inhomogeneous shape"));
                    }
                    inner = Some(s);
                }
                let mut out = vec![items.len()];
                out.extend(inner.unwrap_or_default());
                Ok(out)
            }
            _ => Ok(vec![]),
        }
    }
    fn values(v: &Json, out: &mut Vec<f64>) -> Result<()> {
        match v {
            Json::Arr(items) => items.iter().try_for_each(|x| values(x, out)),
            Json::Null => {
                out.push(f64::NAN);
                Ok(())
            }
            Json::Obj(_) => Err(Error::invalid("float() argument must be a string or a real number, not 'dict'")),
            Json::Str(s) => {
                out.push(float_or_err(s)?);
                Ok(())
            }
            x => {
                out.push(x.float().unwrap_or(f64::NAN));
                Ok(())
            }
        }
    }
    let s = shape(v)?;
    let mut out = Vec::new();
    values(v, &mut out)?;
    Ok((s, out))
}

// ------------------------------------------------------------------------- CSV

/// Records of a CSV file as Python's `csv.reader` (excel dialect) reads it:
/// quoted fields may hold delimiters, doubled quotes and newlines; a blank
/// line is an empty record.
fn csv_records(text: &str) -> Vec<Vec<String>> {
    #[derive(PartialEq, Clone, Copy)]
    enum S {
        StartRecord,
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
        EatCrnl,
    }
    // Lines as a file opened with newline="" yields them: split after \n, \r or \r\n.
    let mut lines = Vec::new();
    let b = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\n' || b[i] == b'\r' {
            let end = if b[i] == b'\r' && b.get(i + 1) == Some(&b'\n') { i + 2 } else { i + 1 };
            lines.push(&text[start..end]);
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }
    if start < b.len() {
        lines.push(&text[start..]);
    }
    let mut records = Vec::new();
    let (mut fields, mut field, mut state) = (Vec::new(), String::new(), S::StartRecord);
    let mut line_iter = lines.into_iter();
    loop {
        let Some(line) = line_iter.next() else {
            if state == S::InQuoted {
                fields.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut fields));
            }
            return records;
        };
        for c in line.chars().map(Some).chain(std::iter::once(None)) {
            let eol = c.is_none();
            let nl = matches!(c, Some('\n' | '\r'));
            if state == S::StartRecord {
                if eol {
                    continue;
                }
                if nl {
                    state = S::EatCrnl;
                    continue;
                }
                state = S::StartField;
            }
            match state {
                S::StartField | S::InField => {
                    if nl || eol {
                        fields.push(std::mem::take(&mut field));
                        state = if eol { S::StartRecord } else { S::EatCrnl };
                    } else if state == S::StartField && c == Some('"') {
                        state = S::InQuoted;
                    } else if c == Some(',') {
                        fields.push(std::mem::take(&mut field));
                        state = S::StartField;
                    } else {
                        field.push(c.unwrap());
                        state = S::InField;
                    }
                }
                S::InQuoted => match c {
                    None => {}
                    Some('"') => state = S::QuoteInQuoted,
                    Some(ch) => field.push(ch),
                },
                S::QuoteInQuoted => match c {
                    Some('"') => {
                        field.push('"');
                        state = S::InQuoted;
                    }
                    Some(',') => {
                        fields.push(std::mem::take(&mut field));
                        state = S::StartField;
                    }
                    _ if nl || eol => {
                        fields.push(std::mem::take(&mut field));
                        state = if eol { S::StartRecord } else { S::EatCrnl };
                    }
                    Some(ch) => {
                        field.push(ch);
                        state = S::InField;
                    }
                    None => unreachable!(),
                },
                S::EatCrnl => {
                    if eol {
                        state = S::StartRecord;
                    }
                }
                S::StartRecord => unreachable!(),
            }
        }
        if state == S::StartRecord {
            records.push(std::mem::take(&mut fields));
        }
    }
}

// ------------------------------------------------------------------------- XML

fn is_el(n: &roxmltree::Node, tag: &str) -> bool {
    n.is_element() && n.tag_name().namespace().is_none() && n.tag_name().name() == tag
}

/// ElementTree's `.text`: the character data before the first child element.
fn et_text(n: roxmltree::Node) -> String {
    let mut s = String::new();
    for c in n.children() {
        if c.is_element() {
            break;
        }
        if c.is_text() {
            s.push_str(c.text().unwrap_or(""));
        }
    }
    s
}

/// ElementTree's `findtext("a/b")`: the text of the first element on the
/// path of child tags ("" if it has none), None if there is none.
fn findtext(n: roxmltree::Node, path: &[&str]) -> Option<String> {
    fn find<'a, 'i>(n: roxmltree::Node<'a, 'i>, path: &[&str]) -> Option<roxmltree::Node<'a, 'i>> {
        let (first, rest) = path.split_first()?;
        n.children().filter(|c| is_el(c, first)).find_map(|c| if rest.is_empty() { Some(c) } else { find(c, rest) })
    }
    find(n, path).map(et_text)
}

/// Decode an XML file: UTF-8 (with or without a byte order mark), UTF-16
/// with a byte order mark, or Latin-1 when the declaration says so.
fn decode_xml(path: &Path, bytes: Vec<u8>) -> Result<String> {
    if let Some(rest) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return String::from_utf8(rest.to_vec()).map_err(|e| Error::file(path, format!("XML: {e}")));
    }
    let utf16 = |be: bool| {
        let units: Vec<u16> = bytes[2..].as_chunks::<2>().0.iter().map(|&c| if be { u16::from_be_bytes(c) } else { u16::from_le_bytes(c) }).collect();
        String::from_utf16(&units).map_err(|e| Error::file(path, format!("XML: {e}")))
    };
    match bytes.get(..2) {
        Some([0xff, 0xfe]) => return utf16(false),
        Some([0xfe, 0xff]) => return utf16(true),
        _ => {}
    }
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]).to_ascii_lowercase();
    let latin1 = head.starts_with("<?xml") && head.split("?>").next().is_some_and(|d| ["iso-8859-1", "latin-1", "latin1", "iso8859-1"].iter().any(|e| d.contains(e)));
    if latin1 {
        return Ok(bytes.iter().map(|&b| b as char).collect());
    }
    String::from_utf8(bytes).map_err(|e| Error::file(path, format!("XML: {e}")))
}

// ----------------------------------------------------------------------- files

/// Entries of a directory whose names pass `keep`, as Python's `Path.glob`
/// lists them (unsorted; nothing for a missing or unreadable directory).
fn glob(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return vec![] };
    entries.flatten().filter(|e| keep(&e.file_name().to_string_lossy())).map(|e| dir.join(e.file_name())).collect()
}

fn is_scan(name: &str) -> bool {
    name.ends_with(".rxp") && !name.contains(".mon.") && !name.contains("residual") && !name.ends_with(".part")
}

/// The `.rxp` scans under a position folder, sorted; monitoring and
/// residual streams excluded. Recursive as `Path.rglob`, which does not
/// follow links to directories.
fn scan_files(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let path = dir.join(e.file_name());
            if is_scan(&e.file_name().to_string_lossy()) {
                out.push(path.clone());
            }
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                walk(&path, out);
            }
        }
    }
    let mut out = Vec::new();
    if recursive {
        walk(dir, &mut out);
    } else {
        out = glob(dir, is_scan);
    }
    out.sort();
    out
}

fn read_dat(root: &Path, name: &str) -> Result<Option<Matrix4>> {
    let dat = root.join("DAT").join(format!("{name}.DAT"));
    if !dat.exists() {
        return Ok(None);
    }
    Ok(Some(Transform::read_matrix_file(&dat)?.to_row_major()))
}

// --------------------------------------------------------------------- parsing

/// A 4x4 matrix from 16 whitespace-separated numbers; None for an empty
/// text or another count.
///
/// # Errors
/// A value Python's `float()` rejects.
pub fn parse_matrix(text: Option<&str>) -> Result<Option<Matrix4>> {
    let Some(text) = text.filter(|t| !t.is_empty()) else { return Ok(None) };
    let vals = py_split(text).map(float_or_err).collect::<Result<Vec<f64>>>()?;
    Ok(vals.try_into().ok())
}

/// `Rz(yaw) @ Ry(pitch) @ Rx(roll)` (degrees) as a 4x4 matrix, row-major.
pub fn rotation_zyx(roll: f64, pitch: f64, yaw: f64) -> Matrix4 {
    let (r, p, y) = (roll.to_radians(), pitch.to_radians(), yaw.to_radians());
    let rx = [[1.0, 0.0, 0.0], [0.0, r.cos(), -r.sin()], [0.0, r.sin(), r.cos()]];
    let ry = [[p.cos(), 0.0, p.sin()], [0.0, 1.0, 0.0], [-p.sin(), 0.0, p.cos()]];
    let rz = [[y.cos(), -y.sin(), 0.0], [y.sin(), y.cos(), 0.0], [0.0, 0.0, 1.0]];
    let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        c
    };
    let m = mul(mul(rz, ry), rx);
    let mut out = [0.0; 16];
    for i in 0..3 {
        out[i * 4..i * 4 + 3].copy_from_slice(&m[i]);
    }
    out[15] = 1.0;
    out
}

/// `project.pop`: the first 16 numbers in the file (None if unreadable or
/// fewer).
fn read_pop(path: &Path) -> Option<Matrix4> {
    let text = read_utf8(path).ok()?;
    // The numbers of `[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?`, leftmost first.
    let b = text.as_bytes();
    let digits = |mut j: usize| {
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let number_at = |i: usize| -> Option<usize> {
        let mut j = i;
        if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
            j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            j = digits(j);
            if j < b.len() && b[j] == b'.' {
                j = digits(j + 1);
            }
        } else if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
            j = digits(j + 1);
        } else {
            return None;
        }
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'-' || b[k] == b'+') {
                k += 1;
            }
            if k < b.len() && b[k].is_ascii_digit() {
                j = digits(k);
            }
        }
        Some(j)
    };
    let mut values = Vec::new();
    let mut i = 0;
    while i < b.len() && values.len() < 16 {
        match number_at(i) {
            Some(j) => {
                values.push(text[i..j].parse::<f64>().ok()?);
                i = j;
            }
            None => i += 1,
        }
    }
    values.try_into().ok()
}

/// `all_sop.csv` (roll, pitch, yaw in degrees and x, y, z) as 4x4 matrices
/// by position name; bad rows are skipped, later rows win.
fn read_all_sop(path: &Path) -> Result<Vec<(String, Matrix4)>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut records = csv_records(&read_utf8(path)?).into_iter();
    let Some(header) = records.next() else { return Ok(vec![]) };
    let mut out: Vec<(String, Matrix4)> = Vec::new();
    for row in records.filter(|r| !r.is_empty()) {
        // dict(zip(header, row)), missing values None, the last repeated key winning.
        let get = |key: &str| -> Option<Option<&str>> {
            header.iter().enumerate().rev().find(|(_, h)| *h == key).map(|(i, _)| row.get(i).map(|s| s.as_str()))
        };
        let name = get("scanPosName").flatten().unwrap_or("").trim_matches(is_py_space).to_string();
        let value = |key: &str| get(key).flatten().and_then(py_float);
        let (Some(x), Some(y), Some(z)) = (value("x"), value("y"), value("z")) else { continue };
        let (Some(roll), Some(pitch), Some(yaw)) = (value("rollDeg"), value("pitchDeg"), value("yawDeg")) else { continue };
        if name.is_empty() || !(x.is_finite() && y.is_finite() && z.is_finite()) {
            continue;
        }
        let mut m = rotation_zyx(roll, pitch, yaw);
        m[3] = x;
        m[7] = y;
        m[11] = z;
        match out.iter_mut().find(|(n, _)| *n == name) {
            Some(entry) => entry.1 = m,
            None => out.push((name, m)),
        }
    }
    Ok(out)
}

fn read_json(path: &Path) -> Option<Json> {
    Json::parse(&read_utf8(path).ok()?)
}

/// The scanner's `level_from_scanner` rotation for one position: from the
/// `.pose` beside the scan (named after its timestamp), `pose_estimation.sop`
/// or `final.pose`, the first that holds a finite `matrix3x3` or roll, pitch
/// and yaw.
///
/// # Errors
/// A `matrix3x3` NumPy could not read as an array.
fn read_attitude(directory: &Path, rxp: Option<&Path>) -> Result<Option<[f64; 9]>> {
    let mut candidates = Vec::new();
    if let Some(rxp) = rxp {
        let name = rxp.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        candidates.push(directory.join(format!("{}.pose", name.split('.').next().unwrap_or(""))));
    }
    candidates.push(directory.join("pose_estimation.sop"));
    candidates.push(directory.join("final.pose"));
    for path in candidates {
        if !path.exists() {
            continue;
        }
        let Some(data) = read_json(&path) else { continue };
        if let Some(matrix) = data.get("matrix3x3").filter(|m| **m != Json::Null) {
            let (shape, values) = json_array(matrix)?;
            if shape == [3, 3] && values.iter().all(|v| v.is_finite()) {
                return Ok(Some(values.try_into().unwrap()));
            }
        }
        let angle = |k: &str| data.get(k).and_then(Json::float);
        let (Some(roll), Some(pitch), Some(yaw)) = (angle("roll"), angle("pitch"), angle("yaw")) else { continue };
        if roll.is_finite() && pitch.is_finite() && yaw.is_finite() {
            let m = rotation_zyx(roll, pitch, yaw);
            return Ok(Some([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]));
        }
    }
    Ok(None)
}

/// The GNSS fix `(latitude, longitude, altitude)` of a scanner `.pose` file,
/// if it has one; a missing altitude is 0.
pub fn read_pose_gnss(path: &Path) -> Option<[f64; 3]> {
    if !path.exists() {
        return None;
    }
    let data = read_json(path)?;
    if !matches!(data, Json::Obj(_)) {
        return None;
    }
    let gnss = data.get("gnss").filter(|g| g.truthy())?;
    let (latitude, longitude) = (gnss.get("latitude")?, gnss.get("longitude")?);
    if *latitude == Json::Null || *longitude == Json::Null {
        return None;
    }
    let altitude = match gnss.get("altitude") {
        Some(a) if a.truthy() => a.float()?,
        _ => 0.0,
    };
    Some([latitude.float()?, longitude.float()?, altitude])
}

fn from_rsp(root: &Path, rsp: &Path) -> Result<RiscanProject> {
    let text = decode_xml(rsp, std::fs::read(rsp)?)?;
    let options = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = roxmltree::Document::parse_with_options(&text, options).map_err(|e| Error::file(rsp, format!("XML: {e}")))?;
    let top = doc.root_element();
    let pop = match top.children().find(|c| is_el(c, "pop")) {
        Some(p) => parse_matrix(findtext(p, &["matrix"]).as_deref())?,
        None => None,
    };
    const KEYS: [&str; 6] = ["theta_start", "theta_delta", "theta_count", "phi_start", "phi_delta", "phi_count"];
    let mut positions = Vec::new();
    for sp in top.descendants().filter(|n| is_el(n, "scanposition")) {
        let name = match sp.attribute("name").filter(|a| !a.is_empty()) {
            Some(a) => a.to_string(),
            None => findtext(sp, &["name"]).unwrap_or_default(),
        };
        let mut sop = parse_matrix(findtext(sp, &["sop", "matrix"]).as_deref())?;
        let pos_dir = root.join("SCANS").join(&name);
        let mut files = if pos_dir.is_dir() { scan_files(&pos_dir, true) } else { vec![] };
        let (mut instrument, mut pattern): (Option<String>, Option<ScanPattern>) = (None, None);
        for scan in sp.descendants().filter(|n| is_el(n, "scan")) {
            if instrument.as_deref().is_none_or(str::is_empty) {
                instrument = findtext(scan, &["instrument"]);
            }
            if pattern.is_none() {
                let vals: Vec<Option<String>> = KEYS.iter().map(|k| findtext(scan, &[*k])).collect();
                if vals.iter().all(|v| v.as_deref().is_some_and(|s| !s.is_empty())) {
                    let v: Vec<&str> = vals.iter().map(|v| v.as_deref().unwrap()).collect();
                    pattern = Some(ScanPattern {
                        theta_start: float_or_err(v[0])?,
                        theta_delta: float_or_err(v[1])?,
                        theta_count: int_or_err(v[2])?,
                        phi_start: float_or_err(v[3])?,
                        phi_delta: float_or_err(v[4])?,
                        phi_count: int_or_err(v[5])?,
                    });
                }
            }
            if let Some(file) = findtext(scan, &["file"]).filter(|f| !f.is_empty()) {
                let candidate = pos_dir.join("SINGLESCANS").join(file);
                if candidate.exists() && !files.contains(&candidate) {
                    files.push(candidate);
                }
            }
        }
        if sop.is_none() {
            sop = read_dat(root, &name)?;
        }
        positions.push(ScanPosition { name, scans: files, sop, instrument, pattern, ..Default::default() });
    }
    let name = findtext(top, &["name"]).filter(|n| !n.is_empty()).unwrap_or_else(|| py_stem(root));
    Ok(RiscanProject { path: root.to_path_buf(), positions, pop, name })
}

fn from_legacy(root: &Path) -> Result<RiscanProject> {
    let pop_file = root.join("project.pop");
    let pop = if pop_file.exists() { read_pop(&pop_file) } else { None };
    let sops = read_all_sop(&root.join("all_sop.csv"))?;
    let sop_of = |name: &str| -> Result<Option<Matrix4>> {
        match sops.iter().find(|(n, _)| n == name) {
            Some((_, m)) => Ok(Some(*m)),
            None => read_dat(root, name),
        }
    };
    let mut positions = Vec::new();
    let mut dirs = glob(&root.join("SCANS"), |n| n.starts_with("ScanPos"));
    dirs.sort();
    for pos_dir in dirs {
        let name = pos_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let scans = scan_files(&pos_dir, true);
        positions.push(ScanPosition { sop: sop_of(&name)?, name, scans, ..Default::default() });
    }
    let mut dirs = glob(root, |n| n.ends_with(".SCNPOS"));
    dirs.sort();
    for pos_dir in dirs {
        // The scanner's own project: the survey scan is in scans/, beside
        // monitoring and tie-point scans that are not survey data.
        let name = py_stem(&pos_dir);
        let scan_dir = pos_dir.join("scans");
        let scans = if scan_dir.is_dir() { scan_files(&scan_dir, false) } else { vec![] };
        let sop = sop_of(&name)?;
        let mut tpl = glob(&pos_dir, |n| n.ends_with(".tpl"));
        tpl.sort();
        let attitude = read_attitude(&pos_dir, scans.first().map(|p| p.as_path()))?;
        positions.push(ScanPosition {
            sop,
            name,
            scans,
            tiepoints: tpl.into_iter().next(),
            gnss: read_pose_gnss(&pos_dir.join("final.pose")),
            attitude,
            ..Default::default()
        });
    }
    Ok(RiscanProject { path: root.to_path_buf(), positions, pop, name: py_stem(root) })
}

/// Parse a RiSCAN PRO project directory (no scan data, only the structure
/// and matrices).
///
/// With `project.rsp`, SOPs and scan patterns come from it (falling back to
/// `DAT/<pos>.DAT`). Without it, the legacy layout is read: `all_sop.csv`,
/// `project.pop` and `SCANS/ScanPos*` or `*.SCNPOS` folders; a scanner
/// `.PROJ` position also gets its attitude, GNSS fix and `.tpl` list.
///
/// # Errors
/// `path` is not a directory; `project.rsp` is not well-formed XML (the
/// message starts with `XML: `); a matrix, pattern or `.DAT` value is not a
/// number.
pub fn read_project(path: impl AsRef<Path>) -> Result<RiscanProject> {
    let root = path.as_ref();
    if !root.is_dir() {
        return Err(Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("not a project directory: {}", root.display()))));
    }
    let rsp = root.join("project.rsp");
    if rsp.exists() {
        from_rsp(root, &rsp)
    } else {
        from_legacy(root)
    }
}

/// A target of a RIEGL `.tpl` tie-point list, in the scanner's frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TiePoint {
    pub position: Point,
    pub reflectance: f64,
    pub diameter: f64,
    pub point_count: i64,
    pub name: String,
}

/// Read a RIEGL `.tpl` tie-point list (JSON); empty for a missing or
/// unreadable file. Entries without a Cartesian position are skipped.
///
/// # Errors
/// An entry whose reflectance, diameter or point count is not a number.
pub fn read_tiepoint_list(path: impl AsRef<Path>) -> Result<Vec<TiePoint>> {
    let Some(Json::Arr(entries)) = read_json(path.as_ref()) else { return Ok(vec![]) };
    let mut out = Vec::new();
    for entry in &entries {
        let Some(c) = entry.get("positionCartesian").filter(|c| c.truthy()) else { continue };
        let coord = |k: &str| c.get(k).and_then(Json::float);
        let (Some(x), Some(y), Some(z)) = (coord("x"), coord("y"), coord("z")) else { continue };
        let number = |k: &str| match entry.get(k) {
            None => Ok(f64::NAN),
            Some(v) => v.float().ok_or_else(|| Error::invalid(format!("{}: {k} is not a number", path.as_ref().display()))),
        };
        let point_count = match entry.get("pointcount") {
            None => 0,
            Some(v) => v.int().ok_or_else(|| Error::invalid(format!("{}: pointcount is not an integer", path.as_ref().display())))?,
        };
        out.push(TiePoint {
            position: [x, y, z],
            reflectance: number("reflectance")?,
            diameter: number("diameter")?,
            point_count,
            name: entry.get("name").map(Json::py_str).unwrap_or_default(),
        });
    }
    Ok(out)
}

// ------------------------------------------------------------------------ GNSS

/// NumPy's pairwise summation (`np.add.reduce` of a contiguous array).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(-0.0, |s, &x| s + x)
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// GNSS fixes `(latitude, longitude, altitude)` as local metres (east,
/// north, altitude) about the survey's own centre, by an equirectangular
/// projection; NaN where there was no fix.
pub fn gnss_to_local(coordinates: &[Option<[f64; 3]>]) -> Vec<[f64; 3]> {
    let mut out = vec![[f64::NAN; 3]; coordinates.len()];
    let known: Vec<[f64; 3]> = coordinates.iter().flatten().copied().collect();
    if known.is_empty() {
        return out;
    }
    let mean = |k: usize| pairwise_sum(&known.iter().map(|c| c[k]).collect::<Vec<_>>()) / known.len() as f64;
    let (lat0, lon0) = (mean(0), mean(1));
    let scale = lat0.to_radians().cos();
    for (o, c) in out.iter_mut().zip(coordinates) {
        if let Some(c) = c {
            *o = [(c[1] - lon0) * 111_320.0 * scale, (c[0] - lat0) * 111_320.0, c[2]];
        }
    }
    out
}

// ------------------------------------------------------------ RiSCAN's filters

/// The attributes a RiSCAN PRO export filter may bound.
pub const EXPORT_ATTRIBUTES: [&str; 4] = ["range", "deviation", "reflectance", "amplitude"];

/// Read a RiSCAN PRO export filter settings file: one `name, minimum,
/// maximum` line per attribute (`;` also separates, `#` starts a comment),
/// names lower-cased without a leading `riegl.`. In file order; a repeated
/// attribute keeps its first place and its last bounds.
///
/// # Errors
/// A malformed line, an unknown attribute, a bound that is not a number or
/// a minimum above its maximum.
pub fn read_export_settings(path: impl AsRef<Path>) -> Result<Vec<(String, f64, f64)>> {
    let path = path.as_ref();
    let text = read_utf8(path)?;
    let mut settings: Vec<(String, f64, f64)> = Vec::new();
    for raw in py_splitlines(&text) {
        let line = raw.split('#').next().unwrap_or("").trim_matches(is_py_space);
        if line.is_empty() {
            continue;
        }
        let replaced = line.replace(';', ",");
        let parts: Vec<&str> = replaced.split(',').map(|p| p.trim_matches(is_py_space)).collect();
        if parts.len() != 3 {
            return Err(Error::invalid(format!("{}: expected 'attribute, min, max', got {}", path.display(), py_repr(raw))));
        }
        let lower = parts[0].to_lowercase();
        let name = lower.strip_prefix("riegl.").unwrap_or(&lower).to_string();
        if !EXPORT_ATTRIBUTES.contains(&name.as_str()) {
            return Err(Error::invalid(format!("{}: unknown attribute {}; known: ['amplitude', 'deviation', 'range', 'reflectance']", path.display(), py_repr(parts[0]))));
        }
        let (lo, hi) = (float_or_err(parts[1])?, float_or_err(parts[2])?);
        if lo > hi {
            return Err(Error::invalid(format!("{}: minimum above maximum for {name}", path.display())));
        }
        match settings.iter_mut().find(|s| s.0 == name) {
            Some(s) => (s.1, s.2) = (lo, hi),
            None => settings.push((name, lo, hi)),
        }
    }
    Ok(settings)
}

/// Keep-mask of points within every closed interval of `settings`, as
/// RiSCAN's export filter: `range` from the scanner's origin, the others
/// from `attribute(name)`, where RIEGL's "not measured" deviation (65535)
/// counts as -1.
///
/// # Errors
/// An attribute the settings bound is missing, or has the wrong length.
pub fn export_settings_mask<'a>(settings: &[(String, f64, f64)], xyz: &[Point], attribute: impl Fn(&str) -> Option<&'a [f64]>) -> Result<Vec<bool>> {
    let mut keep = vec![true; xyz.len()];
    for (name, lo, hi) in settings {
        let inside = |k: &mut bool, v: f64| *k &= v >= *lo && v <= *hi;
        if name == "range" {
            for (k, p) in keep.iter_mut().zip(xyz) {
                inside(k, (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
            }
            continue;
        }
        let values = attribute(name).ok_or_else(|| Error::invalid(format!("export settings need attribute '{name}', which this source lacks")))?;
        if values.len() != xyz.len() {
            return Err(Error::invalid(format!("attribute {name} has {} values for {} points", values.len(), xyz.len())));
        }
        let deviation = name == "deviation";
        for (k, &v) in keep.iter_mut().zip(values) {
            inside(k, if deviation && v == 65535.0 { -1.0 } else { v });
        }
    }
    Ok(keep)
}

/// NumPy's float32 `np.degrees`: `x * (180f / pi_f)`.
fn degrees_f32(x: f32) -> f32 {
    x * (180.0f32 / std::f32::consts::PI)
}

/// NumPy's float32 `x % 360` (the result takes the divisor's sign).
fn mod360_f32(x: f32) -> f32 {
    let m = x % 360.0;
    if m == 0.0 {
        0.0
    } else if m < 0.0 {
        m + 360.0
    } else {
        m
    }
}

/// `np.linalg.norm` of a float32 row.
fn norm_f32(p: &[f32; 3]) -> f32 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

/// `np.median` of float32 values.
fn median_f32(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `np.percentile` (linear) of sorted float32 values, in float64 as NumPy
/// returns it; the neighbours' difference is taken in float32.
fn percentile_f32(sorted: &[f32], q: f64) -> f64 {
    let n = sorted.len();
    let q = q / 100.0;
    let virtual_index = (n as f64 * q + (1.0 - q)) - 1.0;
    if virtual_index >= (n - 1) as f64 {
        return sorted[n - 1] as f64;
    }
    let prev = virtual_index.floor().max(0.0);
    let t = virtual_index - prev;
    let (a, b) = (sorted[prev as usize], sorted[prev as usize + 1]);
    let diff = (b - a) as f64;
    if t >= 0.5 {
        b as f64 - diff * (1.0 - t)
    } else {
        a as f64 + diff * t
    }
}

/// The scan pattern's angular increments `(theta_step, phi_step)` in
/// degrees, estimated from points in the scanner's frame in recording
/// order (float32, as the RXP stream).
///
/// The polar step is the median step between consecutive records; the
/// azimuth step the median over a dozen polar rows of the spacing of the
/// azimuths within a row. Falls back to 0.03 degrees, the usual VZ-series
/// setting, for a scan too small to tell.
///
/// # Errors
/// Every one of the first `sample` points (at least 1000) lies within 5 cm
/// of the scanner.
pub fn angular_steps(xyz: &[[f32; 3]], sample: usize) -> Result<(f64, f64)> {
    let n = xyz.len().min(sample);
    if n < 1000 {
        return Ok((0.03, 0.03));
    }
    let (mut theta, mut phi) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for p in &xyz[..n] {
        let r = norm_f32(p);
        if r > 0.05f32 {
            theta.push(degrees_f32((p[2] / r).clamp(-1.0, 1.0).acos()));
            phi.push(mod360_f32(degrees_f32(p[1].atan2(p[0]))));
        }
    }
    if theta.is_empty() {
        return Err(Error::invalid("index -1 is out of bounds for axis 0 with size 0"));
    }
    let steps: Vec<f32> = theta.windows(2).map(|w| (w[1] - w[0]).abs()).filter(|&s| s > 1e-3f32 && s < 0.5f32).collect();
    let theta_step = if steps.len() > 100 { median_f32(steps) as f64 } else { 0.03 };
    let mut sorted = theta.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (percentile_f32(&sorted, 10.0), percentile_f32(&sorted, 90.0));
    let half = 0.5 * theta_step;
    let (half_f32, step_div) = (half as f32, (hi - lo) / 11.0);
    let mut row_steps = Vec::new();
    for k in 0..12 {
        let centre = if k == 11 { hi } else { k as f64 * step_div + lo };
        let mut row: Vec<f32> = theta.iter().zip(&phi).filter(|(&t, _)| (t as f64 - centre).abs() < half).map(|(_, &p)| p).collect();
        row.sort_by(|a, b| a.total_cmp(b));
        let dphi: Vec<f32> = row.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > half_f32 && d < 1.0f32).collect();
        if dphi.len() >= 50 {
            row_steps.push(median_f32(dphi));
        }
    }
    let phi_step = if row_steps.len() >= 3 { median_f32(row_steps) as f64 } else { 0.03 };
    Ok((theta_step, phi_step))
}

/// Settings of [`riscan_like_mask`]'s `"legacy"` mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegacyFilter {
    pub window_steps: f64,
    pub window_range: f64,
    pub min_neighbours: usize,
    pub weak_db: f64,
    /// `(theta_step, phi_step)` of the scan pattern (degrees); estimated by
    /// [`angular_steps`] if None.
    pub steps: Option<(f64, f64)>,
}

impl Default for LegacyFilter {
    fn default() -> Self {
        LegacyFilter { window_steps: 1.5, window_range: 1.0, min_neighbours: 6, weak_db: 12.0, steps: None }
    }
}

/// What RiSCAN PRO's RXP import keeps, approximately, as a keep-mask over
/// points in the scanner's frame in recording order (float32, as the RXP
/// stream; thresholds are compared in float32 as NumPy compares them).
///
/// `"none"` keeps everything; `"current"` drops `range < min_range`;
/// `"legacy"` also drops an echo weaker than `weak_db` with fewer than
/// `min_neighbours` other echoes within `window_steps` scan increments in
/// either angle and `window_range` metres of range.
///
/// # Errors
/// An unknown mode or an amplitude of another length; see [`angular_steps`].
pub fn riscan_like_mask(xyz: &[[f32; 3]], amplitude: &[f32], mode: &str, min_range: f64, legacy: &LegacyFilter) -> Result<Vec<bool>> {
    if !["none", "current", "legacy"].contains(&mode) {
        return Err(Error::invalid(format!("mode must be one of ('none', 'current', 'legacy'), not {}", py_repr(mode))));
    }
    let n = xyz.len();
    if mode == "none" || n == 0 {
        return Ok(vec![true; n]);
    }
    let r: Vec<f32> = xyz.iter().map(norm_f32).collect();
    let min_range = min_range as f32;
    let mut keep: Vec<bool> = r.iter().map(|&r| r >= min_range).collect();
    if mode == "current" {
        return Ok(keep);
    }
    if amplitude.len() != n {
        return Err(Error::invalid(format!("amplitude has {} values for {n} points", amplitude.len())));
    }
    let (theta_step, phi_step) = match legacy.steps {
        Some(s) => s,
        None => angular_steps(xyz, 2_000_000)?,
    };
    // The unit ball of this space is the window: +-window_steps increments in
    // either angle and +-window_range metres of range.
    let (phi_scale, theta_scale) = ((phi_step * legacy.window_steps) as f32, (theta_step * legacy.window_steps) as f32);
    let range_scale = legacy.window_range as f32;
    let q: Vec<Point> = xyz.iter().zip(&r).map(|(p, &r)| {
        // np.maximum keeps a NaN range.
        let safe = if r.is_nan() { r } else { r.max(1e-6f32) };
        let zenith = degrees_f32((p[2] / safe).clamp(-1.0, 1.0).acos());
        [(mod360_f32(degrees_f32(p[1].atan2(p[0]))) / phi_scale) as f64, (zenith / theta_scale) as f64, (r / range_scale) as f64]
    }).collect();
    let neighbours = filters::count_within(&q, 1.0);
    let weak_db = legacy.weak_db as f32;
    for ((k, &c), &a) in keep.iter_mut().zip(&neighbours).zip(amplitude) {
        if a < weak_db && c.saturating_sub(1) < legacy.min_neighbours {
            *k = false;
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-riscan-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    #[test]
    fn python_numbers() {
        assert_eq!(py_float(" 1_000.5\n"), Some(1000.5));
        assert_eq!(py_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(py_float("nan").unwrap().is_nan());
        for bad in ["", "1__0", "_1", "1_", "0x10", ".", "1e", "1,5"] {
            assert_eq!(py_float(bad), None, "{bad}");
        }
        assert_eq!(py_int(" +2_512 "), Some(2512));
        assert_eq!(py_int("2512.0"), None);
        assert_eq!(py_repr("it's"), "\"it's\"");
    }

    #[test]
    fn python_json() {
        let v = Json::parse(r#" {"a": [1, 2.5, -Infinity, NaN], "a": {"b": "é😀"}, "c": null} "#).unwrap();
        assert_eq!(v.get("a").unwrap().get("b"), Some(&Json::Str("\u{e9}\u{1f600}".into())));
        assert_eq!(v.get("c"), Some(&Json::Null));
        for bad in ["", "[1,]", "01", "{'a': 1}", "[1] x", "\u{feff}[]", "\"a\u{1}\""] {
            assert!(Json::parse(bad).is_none(), "{bad}");
        }
        assert_eq!(json_array(&Json::parse("[[1, true], [null, \"2\"]]").unwrap()).unwrap().0, vec![2, 2]);
        assert!(json_array(&Json::parse("[[1, 2], [3]]").unwrap()).is_err());
    }

    #[test]
    fn python_csv() {
        let r = csv_records("a,\"b\nc\"x,d\r\n\r\n\"e");
        assert_eq!(r, vec![vec!["a".to_string(), "b\ncx".into(), "d".into()], vec![], vec!["e".into()]]);
        assert_eq!(csv_records("a,\"q\"\"\",\n"), vec![vec!["a".to_string(), "q\"".into(), "".into()]]);
    }

    #[test]
    fn stems_and_lines() {
        assert_eq!(py_stem(Path::new("a.RiSCAN")), "a");
        assert_eq!(py_stem(Path::new(".RiSCAN")), ".RiSCAN");
        assert_eq!(py_stem(Path::new("x/..")), "..");
        assert_eq!(py_stem(Path::new(".")), "");
        assert_eq!(py_splitlines("a\r\nb\rc\n\nd"), vec!["a", "b", "c", "", "d"]);
    }

    #[test]
    fn xml_encodings() {
        let p = Path::new("project.rsp");
        let text = "<?xml version=\"1.0\"?><project><name>Caf\u{e9}</name></project>";
        let mut utf16 = vec![0xff, 0xfe];
        utf16.extend(text.encode_utf16().flat_map(|u| u.to_le_bytes()));
        let latin1 = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><project><name>Caf\u{e9}</name></project>".chars().map(|c| c as u8).collect();
        let mut bom = vec![0xef, 0xbb, 0xbf];
        bom.extend(text.as_bytes());
        for bytes in [utf16, latin1, bom, text.as_bytes().to_vec()] {
            let s = decode_xml(p, bytes).unwrap();
            let doc = roxmltree::Document::parse(&s).unwrap();
            assert_eq!(findtext(doc.root_element(), &["name"]).as_deref(), Some("Caf\u{e9}"));
        }
        assert!(decode_xml(p, vec![b'<', 0xff, b'>']).is_err());
        let doc = roxmltree::Document::parse("<a><m> 1 2<!-- c --> 3<?pi x?> 4<![CDATA[ 5]]>&#32;6<b/>7</m></a>").unwrap();
        assert_eq!(findtext(doc.root_element(), &["m"]).as_deref(), Some(" 1 2 3 4 5 6"));
    }

    #[test]
    fn rotation_is_proper() {
        let m = rotation_zyx(10.0, -20.0, 60.0);
        let r = nalgebra::Matrix3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
        assert!((r.determinant() - 1.0).abs() < 1e-12);
        assert!((m[4].atan2(m[0]).to_degrees() - 60.0).abs() < 25.0);
        assert_eq!(&m[12..], &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(rotation_zyx(0.0, 0.0, 0.0), Transform::identity().to_row_major());
    }

    #[test]
    fn reads_an_rsp_project() {
        let root = tmp("rsp").join("Demo.RiSCAN");
        touch(&root.join("SCANS/ScanPos001/SINGLESCANS/s1.rxp"));
        touch(&root.join("SCANS/ScanPos001/SINGLESCANS/s1.mon.rxp"));
        std::fs::create_dir_all(root.join("DAT")).unwrap();
        std::fs::write(root.join("DAT/ScanPos002.DAT"), "1 0 0 5\n0 1 0 6\n0 0 1 7\n0 0 0 1\n").unwrap();
        std::fs::write(root.join("project.rsp"), r#"<?xml version="1.0"?>
<!DOCTYPE project SYSTEM "./project.dtd" [ ]>
<project><name>Demo</name><pop><matrix> 1 0 0 100 0 1 0 200 0 0 1 300 0 0 0 1</matrix></pop>
<scanpositions><scanposition name="ScanPos001"><singlescans><scan><file>s1.rxp</file><instrument>VZ-400i</instrument>
<theta_start>30</theta_start><theta_delta>0.04</theta_delta><theta_count>2512</theta_count>
<phi_start>0</phi_start><phi_delta>0.04</phi_delta><phi_count>9001</phi_count></scan></singlescans>
<sop><matrix>0 -1 0 1 1 0 0 2 0 0 1 3 0 0 0 1</matrix></sop></scanposition>
<scanposition><name>ScanPos002</name></scanposition></scanpositions></project>"#).unwrap();
        let p = read_project(&root).unwrap();
        assert_eq!(p.name, "Demo");
        assert_eq!(p.pop.unwrap()[3], 100.0);
        assert_eq!(p.positions.len(), 2);
        let a = &p.positions[0];
        assert_eq!(a.scans, vec![root.join("SCANS/ScanPos001/SINGLESCANS/s1.rxp")]);
        assert_eq!(a.instrument.as_deref(), Some("VZ-400i"));
        assert_eq!(a.pattern.unwrap().theta_count, 2512);
        assert_eq!(a.sop.unwrap()[3], 1.0);
        assert_eq!(p.positions[1].sop.unwrap()[11], 7.0);
        assert_eq!(p.with_scans(true), vec![0]);
        std::fs::write(root.join("project.rsp"), "<project>").unwrap();
        assert!(read_project(&root).unwrap_err().to_string().contains("XML: "));
        assert!(read_project(root.join("nothing")).is_err());
    }

    #[test]
    fn reads_a_scanner_project() {
        let root = tmp("proj").join("survey.PROJ");
        for k in 0..2 {
            let pos = root.join(format!("ScanPos00{}.SCNPOS", k + 1));
            touch(&pos.join(format!("scans/26080{k}_120000.rxp")));
            touch(&pos.join(format!("scans/26080{k}_120000.mon.rxp")));
            std::fs::write(pos.join("final.pose"), format!(r#"{{"gnss": {{"latitude": {}, "longitude": 130.8, "altitude": null}}}}"#, -12.5 + k as f64 * 2e-4)).unwrap();
        }
        let pos = root.join("ScanPos002.SCNPOS");
        std::fs::write(pos.join("260801_120000.pose"), r#"{"roll": -70.0, "pitch": -80.0, "yaw": 150.0}"#).unwrap();
        touch(&pos.join("260801_120000.tpl"));
        std::fs::write(root.join("all_sop.csv"), "scanPosName,x,y,z,rollDeg,pitchDeg,yawDeg\nScanPos001,0,0,0,0,0,0\nScanPos002,20,5,0.1,0.5,-0.25,30\n").unwrap();
        let p = read_project(&root).unwrap();
        assert_eq!(p.name, "survey");
        assert_eq!(p.positions.iter().map(|q| q.name.as_str()).collect::<Vec<_>>(), ["ScanPos001", "ScanPos002"]);
        let b = &p.positions[1];
        assert_eq!(b.rxp().unwrap().file_name().unwrap(), "260801_120000.rxp");
        assert_eq!((b.sop.unwrap()[3], b.sop.unwrap()[7]), (20.0, 5.0));
        assert!(b.attitude.unwrap()[8] < 0.5, "a tilted scanner's z-axis is not up");
        assert!(p.positions[0].attitude.is_none());
        assert_eq!(b.gnss.unwrap()[2], 0.0);
        assert!(b.tiepoints.is_some() && p.positions[0].tiepoints.is_none());
        let local = gnss_to_local(&p.positions.iter().map(|q| q.gnss).collect::<Vec<_>>());
        assert!((local[1][1] - local[0][1] - 22.264).abs() < 0.01);
        assert!(gnss_to_local(&[None])[0][0].is_nan());
    }

    #[test]
    fn export_settings() {
        let d = tmp("settings");
        let f = d.join("s.txt");
        std::fs::write(&f, "deviation, 0, 12\nriegl.Range; 2; 100 # comment\nreflectance, -20, 5\n").unwrap();
        let s = read_export_settings(&f).unwrap();
        assert_eq!(s, vec![("deviation".into(), 0.0, 12.0), ("range".into(), 2.0, 100.0), ("reflectance".into(), -20.0, 5.0)]);
        let xyz = [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [200.0, 0.0, 0.0]];
        let (dev, refl) = ([1.0, 1.0, 20.0, 65535.0, 1.0], [-5.0, -5.0, -5.0, -5.0, -5.0]);
        let get = |n: &str| match n {
            "deviation" => Some(&dev[..]),
            "reflectance" => Some(&refl[..]),
            _ => None,
        };
        assert_eq!(export_settings_mask(&s, &xyz, get).unwrap(), [false, true, false, false, false]);
        assert!(export_settings_mask(&s, &xyz, |_| None).is_err());
        std::fs::write(&f, "colour, 0, 1\n").unwrap();
        assert!(read_export_settings(&f).is_err());
        std::fs::write(&f, "range, 5, 1\n").unwrap();
        assert!(read_export_settings(&f).is_err());
    }

    fn wall(step: f32) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for l in 0..40 {
            for s in 0..200 {
                let (t, p) = ((80.0 + step * s as f32).to_radians(), (step * l as f32).to_radians());
                out.push([10.0 * t.sin() * p.cos(), 10.0 * t.sin() * p.sin(), 10.0 * t.cos()]);
            }
        }
        out
    }

    #[test]
    fn steps_and_filters() {
        let w = wall(0.03);
        let (t, p) = angular_steps(&w, 2_000_000).unwrap();
        assert!((t - 0.03).abs() < 0.003 && (p - 0.03).abs() < 0.003, "{t} {p}");
        assert_eq!(angular_steps(&w[..999], 2_000_000).unwrap(), (0.03, 0.03));
        assert!(angular_steps(&vec![[0.0; 3]; 2000], 2_000_000).is_err());
        let mut xyz = w.clone();
        xyz.extend([[0.1, 0.0, 0.0], [0.3, 0.1, 0.0], [0.7, 0.0, 0.0]]);
        let amp = vec![15.0f32; xyz.len()];
        let keep = riscan_like_mask(&xyz, &amp, "current", 0.5, &LegacyFilter::default()).unwrap();
        assert!(keep[..w.len()].iter().all(|&k| k) && keep[w.len()..] == [false, false, true]);
        let mut xyz = w.clone();
        xyz.extend(w[..6].iter().map(|p| [p[0] * 0.7, p[1] * 0.7, p[2] * 0.7]));
        let mut amp = vec![15.0f32; xyz.len()];
        amp[w.len()..w.len() + 3].fill(5.0);
        let legacy = LegacyFilter { steps: Some((0.03, 0.03)), ..Default::default() };
        let keep = riscan_like_mask(&xyz, &amp, "legacy", 0.5, &legacy).unwrap();
        assert!(keep[..w.len()].iter().all(|&k| k));
        assert_eq!(keep[w.len()..], [false, false, false, true, true, true]);
        assert!(riscan_like_mask(&xyz, &amp, "strict", 0.5, &legacy).is_err());
        assert!(riscan_like_mask(&xyz, &amp, "none", 0.5, &legacy).unwrap().iter().all(|&k| k));
    }

    #[test]
    fn numpy_helpers() {
        let v: Vec<f64> = (0..300).map(|i| 153.0 + (i as f64 * 0.37).sin() * 1e-3).collect();
        let naive: f64 = v.iter().sum();
        assert!((pairwise_sum(&v) - naive).abs() < 1e-9);
        assert_eq!(mod360_f32(-30.0), 330.0);
        assert_eq!(mod360_f32(720.0), 0.0);
        assert_eq!(percentile_f32(&[0.0, 1.0, 2.0, 3.0], 10.0), 0.30000000000000004);
        assert_eq!(parse_matrix(Some("1 2 3")).unwrap(), None);
        assert!(parse_matrix(Some("1 x")).is_err());
    }

    #[test]
    fn tiepoint_lists() {
        let d = tmp("tpl");
        let f = d.join("a.tpl");
        std::fs::write(&f, r#"[{"positionCartesian": {"x": 1, "y": "2", "z": 3.5}, "reflectance": -2, "name": "T1", "pointcount": 12.9},
                               {"positionCartesian": {"x": 1}}, 5, {"positionCartesian": null}]"#).unwrap();
        let t = read_tiepoint_list(&f).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].position, [1.0, 2.0, 3.5]);
        assert_eq!((t[0].point_count, t[0].name.as_str()), (12, "T1"));
        assert!(t[0].diameter.is_nan());
        assert!(read_tiepoint_list(d.join("missing.tpl")).unwrap().is_empty());
    }

    #[test]
    fn json_reads_what_the_json_module_reads() {
        use crate::json::{self as j, Json as J};
        // The same documents through this parser and crate::json, which is checked against Python.
        fn same(a: &Json, b: &J) -> bool {
            match (a, b) {
                (Json::Null, J::Null) => true,
                (Json::Bool(x), J::Bool(y)) => x == y,
                (Json::Num(x, true), J::Int(y)) => *x == *y as f64,
                (Json::Num(x, false), J::Float(y)) => x == y || (x.is_nan() && y.is_nan()),
                (Json::Str(x), J::Str(y)) => x == y,
                (Json::Arr(x), J::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
                (Json::Obj(x), J::Object(y)) => x.len() == y.len() && x.iter().zip(y).all(|((k, p), (l, q))| k == l && same(p, q)),
                _ => false,
            }
        }
        let docs = [
            r#"{"a": [1, -2.5e3, true, false, null], "b": {"c": "x\"y\\z\/\b\f\n\r\t\u00e9"}}"#,
            r#"[NaN, Infinity, -Infinity, 0, -0.0, 1E2]"#,
            r#""\ud83d\ude00 and \ud83d alone""#,
            r#"{"k": 1, "k": 2}"#,
            "  [ ]  ",
        ];
        for d in docs {
            let (a, b) = (Json::parse(d).unwrap(), j::parse(d).unwrap());
            assert!(same(&a, &b), "{d}: {a:?} vs {b:?}");
        }
        assert_eq!(Json::parse(r#"{"k": 1, "k": 2}"#).unwrap().get("k"), Some(&Json::Num(2.0, true)));
        for bad in ["[1,]", "{\"a\" 1}", "{1: 2}", "[01]", "\"\\q\"", "\"\\u12G4\"", "\"ab", "[1 2]", "tru", "\"a\x01\"", "[1] x", ""] {
            assert!(Json::parse(bad).is_none(), "{bad:?}");
            assert!(j::parse(bad).is_err(), "{bad:?}");
        }
        // Python's bool(), int(), float() and str() of the decoded values.
        let v = |t: &str| Json::parse(t).unwrap();
        for (t, truth) in [("\"\"", false), ("\"a\"", true), ("0", false), ("0.5", true), ("[]", false), ("{\"a\": 1}", true), ("false", false), ("null", false)] {
            assert_eq!(v(t).truthy(), truth, "{t}");
        }
        assert_eq!((v("true").int(), v("-3.9").int(), v("\" 12 \"").int(), v("\"1.5\"").int(), v("NaN").int(), v("[]").int()), (Some(1), Some(-3), Some(12), None, None, None));
        assert_eq!((v("false").float(), v("\" 2.5\"").float(), v("{}").float()), (Some(0.0), Some(2.5), None));
        let strs: Vec<String> = ["null", "true", "false", "3", "2.5", "NaN", "-Infinity", "\"ab\""].iter().map(|t| v(t).py_str()).collect();
        assert_eq!(strs, ["None", "True", "False", "3", "2.5", "nan", "-inf", "ab"]);
    }
}
