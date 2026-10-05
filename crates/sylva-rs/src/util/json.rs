// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A small JSON reader and writer for the files Sylva exchanges (RIEGL
//! tie-point lists, stem maps).
//!
//! The reader accepts what Python's `json.loads` accepts, `NaN` and
//! `Infinity` included. The writer lays values out as `json.dumps(value,
//! indent=2)` does, floats in Python's shortest round-trip notation, so a
//! file written here and one written by the Python package are the same
//! bytes.

use crate::error::{Error, Result};

/// A JSON value. Objects keep their key order; numbers remember whether
/// they were written as integers.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value of `key` in an object (the last one, as Python keeps).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python's truth value of the decoded object.
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(i) => *i != 0,
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
        }
    }
}

/// Parse a JSON document.
pub fn parse(text: &str) -> Result<Json> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.fail("extra data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn fail(&self, what: &str) -> Error {
        Error::invalid(format!("JSON: {what} at byte {}", self.i))
    }

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

    fn value(&mut self) -> Result<Json> {
        let Some(&c) = self.s.get(self.i) else { return Err(self.fail("expecting value")) };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' if self.eat("true") => Ok(Json::Bool(true)),
            b'f' if self.eat("false") => Ok(Json::Bool(false)),
            b'n' if self.eat("null") => Ok(Json::Null),
            b'N' if self.eat("NaN") => Ok(Json::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Ok(Json::Float(f64::INFINITY)),
            b'-' if self.eat("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.fail("expecting value")),
        }
    }

    fn object(&mut self) -> Result<Json> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("}") {
            return Ok(Json::Object(items));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.fail("expecting property name"));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.fail("expecting ':'"));
            }
            self.ws();
            let v = self.value()?;
            items.push((k, v));
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(Json::Object(items));
            }
            return Err(self.fail("expecting ',' or '}'"));
        }
    }

    fn array(&mut self) -> Result<Json> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("]") {
                return Ok(Json::Array(items));
            }
            return Err(self.fail("expecting ',' or ']'"));
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.fail("bad \\u escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| self.fail("bad \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| self.fail("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && !matches!(self.s[self.i], b'"' | b'\\') && self.s[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).map_err(|_| self.fail("invalid UTF-8"))?);
            match self.s.get(self.i) {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else { return Err(self.fail("unterminated string")) };
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
                            let mut c = self.hex4()?;
                            if (0xd800..0xdc00).contains(&c) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    c = 0x10000 + ((c - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.fail("invalid escape")),
                    }
                }
                _ => return Err(self.fail("unterminated string")),
            }
        }
    }

    fn number(&mut self) -> Result<Json> {
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
        let int_start = self.i;
        let n = digits(self);
        if n == 0 || (n > 1 && self.s[int_start] == b'0') {
            self.i = start;
            return Err(self.fail("expecting value"));
        }
        let mut float = false;
        if self.s.get(self.i) == Some(&b'.') && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
            digits(self);
            float = true;
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
                float = true;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII");
        if !float {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Json::Int(v));
            }
        }
        Ok(Json::Float(text.parse::<f64>().map_err(|_| self.fail("bad number"))?))
    }
}

/// Python's `repr` of a float: the shortest string that reads back to the
/// same value, positional from 1e-4 up to 1e16 and scientific otherwise.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` gives the shortest round-trip digits.
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let n = digits.len() as i32;
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point >= n {
            format!("{}{}.0", digits, "0".repeat((point - n) as usize))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let m = if n > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_value(out: &mut String, v: &Json, indent: usize, level: usize) {
    let pad = |out: &mut String, level: usize| {
        out.push('\n');
        out.push_str(&" ".repeat(indent * level));
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => out.push_str(&if f.is_nan() {
            "NaN".to_string()
        } else if f.is_infinite() {
            if *f > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
        } else {
            py_float_repr(*f)
        }),
        Json::Str(s) => write_str(out, s),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                write_value(out, item, indent, level + 1);
            }
            pad(out, level);
            out.push(']');
        }
        Json::Object(items) => {
            if items.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (k, (key, item)) in items.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                write_str(out, key);
                out.push_str(": ");
                write_value(out, item, indent, level + 1);
            }
            pad(out, level);
            out.push('}');
        }
    }
}

/// `json.dumps(value, indent=indent)`.
pub fn to_string_indented(v: &Json, indent: usize) -> String {
    let mut out = String::new();
    write_value(&mut out, v, indent, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_read_as_python_writes_them() {
        for (x, s) in [(1.0, "1.0"), (0.1, "0.1"), (1e16, "1e+16"), (1.5e-5, "1.5e-05"), (123456789.25, "123456789.25"), (0.0001, "0.0001"), (-2.5e-7, "-2.5e-07"), (1e22, "1e+22"), (9999999999999998.0, "9999999999999998.0"), (-0.0, "-0.0")] {
            assert_eq!(py_float_repr(x), s);
        }
    }

    #[test]
    fn round_trip_and_layout() {
        let text = "{\n  \"name\": \"a\\u00e9\",\n  \"stems\": [\n    {\n      \"x\": 1.0,\n      \"axis\": [\n        0.0,\n        1e-05\n      ],\n      \"n\": 3,\n      \"e\": []\n    }\n  ]\n}";
        let v = parse(text).unwrap();
        assert_eq!(to_string_indented(&v, 2), text);
        assert_eq!(v.get("name"), Some(&Json::Str("a\u{e9}".into())));
        assert!(parse("[1, 2,").is_err());
        assert!(parse("").is_err());
        let Json::Array(v) = parse(" [NaN, -Infinity, 1e3, -0] ").unwrap() else { panic!("array") };
        assert!(matches!(v[0], Json::Float(f) if f.is_nan()));
        assert_eq!(&v[1..], &[Json::Float(f64::NEG_INFINITY), Json::Float(1000.0), Json::Int(0)]);
    }

    /// `json.dumps(v, indent=2)` of Python 3.14 for the value built in the test.
    const PYTHON: &str = r#"{
  "s": "q\"b\\s/\n\r\t\b\f\u0001 \u00e9 \ud83d\ude00 ~",
  "nan": NaN,
  "inf": [
    Infinity,
    -Infinity
  ],
  "e": {},
  "a": [],
  "t": true,
  "f": false,
  "n": null,
  "neg": -7,
  "x": 2.5e-300
}"#;

    #[test]
    fn writes_escapes_and_specials_as_python_does() {
        let item = |k: &str, v: Json| (k.to_string(), v);
        let v = Json::Object(vec![
            item("s", Json::Str("q\"b\\s/\n\r\t\u{8}\u{c}\u{1} \u{e9} \u{1f600} ~".into())),
            item("nan", Json::Float(f64::NAN)),
            item("inf", Json::Array(vec![Json::Float(f64::INFINITY), Json::Float(f64::NEG_INFINITY)])),
            item("e", Json::Object(vec![])),
            item("a", Json::Array(vec![])),
            item("t", Json::Bool(true)),
            item("f", Json::Bool(false)),
            item("n", Json::Null),
            item("neg", Json::Int(-7)),
            item("x", Json::Float(2.5e-300)),
        ]);
        assert_eq!(to_string_indented(&v, 2), PYTHON);
        let back = parse(PYTHON).unwrap();
        assert_eq!(to_string_indented(&back, 2), PYTHON);
        assert_eq!(back.get("s"), v.get("s"));
        assert_eq!(parse(r#""\/\u0041""#).unwrap(), Json::Str("/A".into()));
    }

    #[test]
    fn truth_and_lookup_as_in_python() {
        // bool(json.loads(text)) in Python.
        for (text, truth) in [("\"\"", false), ("\"x\"", true), ("0", false), ("0.0", false), ("-0.5", true), ("[]", false), ("{}", false), ("[0]", true), ("{\"a\": 0}", true), ("null", false), ("false", false), ("true", true), ("3", true)] {
            assert_eq!(parse(text).unwrap().truthy(), truth, "{text}");
        }
        // Python keeps the last of repeated keys.
        let v = parse(r#"{"a": 1, "b": [2], "a": 3}"#).unwrap();
        assert_eq!(v.get("a"), Some(&Json::Int(3)));
        assert_eq!(v.get("c"), None);
        assert_eq!(Json::Array(vec![]).get("a"), None);
    }

    #[test]
    fn numbers_as_python_reads_them() {
        assert_eq!(parse("1E+2").unwrap(), Json::Float(100.0));
        assert_eq!(parse("-12").unwrap(), Json::Int(-12));
        assert_eq!(parse("2.50").unwrap(), Json::Float(2.5));
        // Beyond i64 Python keeps an int; here it becomes the nearest float.
        assert_eq!(parse("12345678901234567890").unwrap(), Json::Float(12345678901234567890.0));
    }

    #[test]
    fn errors_say_what_and_where() {
        // The same byte offsets as Python's JSONDecodeError.pos for these.
        let err = |text: &str| parse(text).unwrap_err().to_string();
        assert_eq!(err("[1] x"), "JSON: extra data at byte 4");
        assert_eq!(err("1.5e"), "JSON: extra data at byte 3");
        assert_eq!(err(r#"{"a" 1}"#), "JSON: expecting ':' at byte 5");
        assert_eq!(err("{1: 2}"), "JSON: expecting property name at byte 1");
        assert_eq!(err("-"), "JSON: expecting value at byte 0");
        assert_eq!(err("tru"), "JSON: expecting value at byte 0");
        assert_eq!(err("[1 2]"), "JSON: expecting ',' or ']' at byte 3");
        assert_eq!(err(r#"{"a": 1 "b": 2}"#), "JSON: expecting ',' or '}' at byte 8");
        // Python reports these at the start of the string or escape instead.
        assert!(err(r#""abc"#).starts_with("JSON: unterminated string"));
        assert!(err("\"ab\\").starts_with("JSON: unterminated string"));
        assert!(err(r#""\q""#).starts_with("JSON: invalid escape"));
        assert!(err(r#""\u12""#).starts_with("JSON: bad \\u escape"));
        assert!(err(r#"{"a": 1,}"#).starts_with("JSON: expecting property name"));
        assert!(parse("[01]").is_err());
        assert!(parse("\"a\u{1}b\"").is_err(), "control characters must be escaped");
    }

    #[test]
    fn surrogate_pairs_join_and_lone_surrogates_are_replaced() {
        assert_eq!(parse(r#""\ud83d\ude00""#).unwrap(), Json::Str("\u{1f600}".into()));
        // Python keeps a lone surrogate; a Rust string cannot hold one.
        assert_eq!(parse(r#""\ud83d x""#).unwrap(), Json::Str("\u{fffd} x".into()));
        assert_eq!(parse(r#""\ud83d\u0041""#).unwrap(), Json::Str("\u{fffd}A".into()));
    }
}
