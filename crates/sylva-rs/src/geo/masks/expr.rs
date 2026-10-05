// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A small expression language for attribute masks.
//!
//! ```text
//! height > 2 & classification != 2
//! 1.3 <= z < 40 and not (classification in (7, 18))
//! intensity / 65535 > 0.2 | withheld
//! ```
//!
//! Grammar, loosest binding first:
//!
//! | Level | Syntax |
//! |---|---|
//! | or | `a | b`, `a || b`, `a or b` |
//! | and | `a & b`, `a && b`, `a and b` |
//! | not | `!a`, `not a` |
//! | comparison | `a < b`, `<=`, `>`, `>=`, `==`, `!=`; chains such as `0 < z <= 5`; `a in (1, 2, 3)`, `a not in (...)` |
//! | sum | `a + b`, `a - b` |
//! | product | `a * b`, `a / b` |
//! | sign | `-a`, `+a` |
//! | atom | a number, `true`, `false`, `x`, `y`, `z`, an attribute name, `( ... )` |
//!
//! `x`, `y` and `z` are the coordinates (an attribute with one of those names
//! is hidden). Values are compared as float64, with IEEE semantics: a
//! comparison with NaN is false except `!=`, which is true; division by zero
//! gives an infinity or NaN. A boolean attribute is a condition in its own
//! right and counts as 0 or 1 in arithmetic; a numeric value used as a
//! condition is an error. The expression is only parsed and evaluated,
//! never executed as code.

use std::collections::BTreeMap;
use std::fmt::Display;

use rayon::prelude::*;

use super::CHUNK;
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::Point;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Num(f64),
    Bool(bool),
    Var(String),
    Neg(Box<Node>),
    Not(Box<Node>),
    Arith(ArOp, Box<Node>, Box<Node>),
    Cmp(CmpOp, Box<Node>, Box<Node>),
    And(Box<Node>, Box<Node>),
    Or(Box<Node>, Box<Node>),
    In(Box<Node>, Vec<f64>, bool),
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    kind: Kind,
    /// Character offset of the node's first token in the source.
    pos: usize,
}

/// A parsed mask expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    src: String,
    root: Node,
}

fn marked(src: &str, pos: usize, msg: impl Display) -> String {
    format!("{msg}\n    {src}\n    {}^", " ".repeat(pos))
}

fn syntax(src: &str, pos: usize, msg: impl Display) -> Error {
    Error::invalid(marked(src, pos, format!("syntax error at position {pos}: {msg}")))
}

// ------------------------------------------------------------------ lexing

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(&'static str),
    End,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    pos: usize,
}

const OPS: [&str; 20] = ["<=", ">=", "==", "!=", "&&", "||", "<", ">", "&", "|", "!", "+", "-", "*", "/", "(", ")", ",", "=", "~"];

/// Split the expression into tokens: numbers, names and operators.
///
/// Each token remembers where it began, which is what lets an error point at
/// the offending character. The text is collected into a `Vec<char>` first so
/// that `i` can step character by character; indexing a `String` directly is
/// not allowed in Rust, because its characters are not all the same width.
fn lex(src: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        // A number: digits, optionally a decimal point, optionally an exponent.
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    i = j;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let text: String = chars[start..i].iter().collect();
            let v: f64 = text.parse().map_err(|_| syntax(src, start, format!("invalid number {text:?}")))?;
            if i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                return Err(syntax(src, i, format!("unexpected {:?} after the number {text}", chars[i])));
            }
            out.push(Token { tok: Tok::Num(v), pos: start });
            continue;
        }
        // A name: an attribute, a coordinate, or a word operator such as
        // `and`. Which it is, is the parser's business, not the lexer's.
        if c.is_alphabetic() || c == '_' {
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Token { tok: Tok::Ident(chars[start..i].iter().collect()), pos: start });
            continue;
        }
        // An operator, two characters before one, so `<=` is not read as `<`.
        let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        if let Some(op) = OPS.iter().find(|op| op.len() == 2 && **op == two) {
            out.push(Token { tok: Tok::Op(op), pos: start });
            i += 2;
            continue;
        }
        if let Some(op) = OPS.iter().find(|op| op.len() == 1 && op.starts_with(c)) {
            match *op {
                "=" => return Err(syntax(src, start, "use '==' to test equality")),
                "~" => return Err(syntax(src, start, "use '!' or 'not' for negation")),
                _ => {}
            }
            out.push(Token { tok: Tok::Op(op), pos: start });
            i += 1;
            continue;
        }
        return Err(syntax(src, start, format!("unexpected character {c:?}")));
    }
    out.push(Token { tok: Tok::End, pos: chars.len() });
    Ok(out)
}

// ----------------------------------------------------------------- parsing

/// Recursive descent over the tokens: one method per level of the grammar
/// above, each calling the level that binds more tightly, so `or` calls `and`
/// calls `not` and so on down to `atom`. `i` is how far along `toks` we are.
///
/// The `<'a>` is a lifetime: it marks that the parser borrows the expression
/// text rather than owning a copy, and so cannot outlive it. It carries no
/// run-time cost and no behaviour — it is a note to the compiler.
struct Parser<'a> {
    src: &'a str,
    toks: Vec<Token>,
    i: usize,
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Num(v) => format!("the number {v}"),
        Tok::Ident(s) => format!("{s:?}"),
        Tok::Op(o) => format!("'{o}'"),
        Tok::End => "the end of the expression".into(),
    }
}

impl Parser<'_> {
    fn peek(&self) -> &Token {
        &self.toks[self.i]
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.i + k).min(self.toks.len() - 1)].tok
    }

    fn is_op(&self, ops: &[&str]) -> Option<&'static str> {
        match self.peek().tok {
            Tok::Op(o) if ops.contains(&o) => Some(o),
            _ => None,
        }
    }

    fn is_word(&self, w: &str) -> bool {
        matches!(&self.peek().tok, Tok::Ident(s) if s == w)
    }

    fn err(&self, msg: impl Display) -> Error {
        syntax(self.src, self.peek().pos, msg)
    }

    fn expect_op(&mut self, op: &str, what: &str) -> Result<()> {
        if self.is_op(&[op]).is_some() {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected {what}, found {}", describe(&self.peek().tok))))
        }
    }

    fn or(&mut self) -> Result<Node> {
        let mut left = self.and()?;
        while self.is_op(&["|", "||"]).is_some() || self.is_word("or") {
            self.i += 1;
            let right = self.and()?;
            let pos = left.pos;
            left = Node { kind: Kind::Or(Box::new(left), Box::new(right)), pos };
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Node> {
        let mut left = self.not()?;
        while self.is_op(&["&", "&&"]).is_some() || self.is_word("and") {
            self.i += 1;
            let right = self.not()?;
            let pos = left.pos;
            left = Node { kind: Kind::And(Box::new(left), Box::new(right)), pos };
        }
        Ok(left)
    }

    fn not(&mut self) -> Result<Node> {
        if self.is_op(&["!"]).is_some() || self.is_word("not") {
            let pos = self.peek().pos;
            self.i += 1;
            let inner = self.not()?;
            return Ok(Node { kind: Kind::Not(Box::new(inner)), pos });
        }
        self.comparison()
    }

    fn cmp_op(&self) -> Option<CmpOp> {
        Some(match self.is_op(&["<", "<=", ">", ">=", "==", "!="])? {
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            "==" => CmpOp::Eq,
            _ => CmpOp::Ne,
        })
    }

    fn comparison(&mut self) -> Result<Node> {
        let first = self.sum()?;
        let pos = first.pos;
        if self.is_word("in") || (self.is_word("not") && matches!(self.peek_at(1), Tok::Ident(s) if s == "in")) {
            let negate = self.is_word("not");
            self.i += if negate { 2 } else { 1 };
            let set = self.number_list()?;
            return Ok(Node { kind: Kind::In(Box::new(first), set, negate), pos });
        }
        let mut result: Option<Node> = None;
        let mut left = first;
        while let Some(op) = self.cmp_op() {
            self.i += 1;
            let right = self.sum()?;
            let cmp = Node { kind: Kind::Cmp(op, Box::new(left), Box::new(right.clone())), pos };
            result = Some(match result {
                None => cmp,
                Some(prev) => Node { kind: Kind::And(Box::new(prev), Box::new(cmp)), pos },
            });
            left = right;
        }
        Ok(result.unwrap_or(left))
    }

    fn number_list(&mut self) -> Result<Vec<f64>> {
        self.expect_op("(", "'(' to start the list after 'in'")?;
        let mut vals = Vec::new();
        loop {
            let sign = if self.is_op(&["-"]).is_some() {
                self.i += 1;
                -1.0
            } else {
                if self.is_op(&["+"]).is_some() {
                    self.i += 1;
                }
                1.0
            };
            match self.peek().tok {
                Tok::Num(v) => {
                    vals.push(sign * v);
                    self.i += 1;
                }
                ref t => return Err(self.err(format!("expected a number in the 'in' list, found {}", describe(t)))),
            }
            if self.is_op(&[","]).is_some() {
                self.i += 1;
                if self.is_op(&[")"]).is_some() {
                    break;
                }
                continue;
            }
            break;
        }
        self.expect_op(")", "',' or ')' in the 'in' list")?;
        Ok(vals)
    }

    fn sum(&mut self) -> Result<Node> {
        let mut left = self.product()?;
        while let Some(o) = self.is_op(&["+", "-"]) {
            self.i += 1;
            let right = self.product()?;
            let pos = left.pos;
            let op = if o == "+" { ArOp::Add } else { ArOp::Sub };
            left = Node { kind: Kind::Arith(op, Box::new(left), Box::new(right)), pos };
        }
        Ok(left)
    }

    fn product(&mut self) -> Result<Node> {
        let mut left = self.sign()?;
        while let Some(o) = self.is_op(&["*", "/"]) {
            self.i += 1;
            let right = self.sign()?;
            let pos = left.pos;
            let op = if o == "*" { ArOp::Mul } else { ArOp::Div };
            left = Node { kind: Kind::Arith(op, Box::new(left), Box::new(right)), pos };
        }
        Ok(left)
    }

    fn sign(&mut self) -> Result<Node> {
        if let Some(o) = self.is_op(&["-", "+"]) {
            let pos = self.peek().pos;
            self.i += 1;
            let inner = self.sign()?;
            return Ok(if o == "-" { Node { kind: Kind::Neg(Box::new(inner)), pos } } else { inner });
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Node> {
        let Token { tok, pos } = self.peek().clone();
        match tok {
            Tok::Num(v) => {
                self.i += 1;
                Ok(Node { kind: Kind::Num(v), pos })
            }
            Tok::Ident(s) => match s.as_str() {
                "true" | "false" => {
                    self.i += 1;
                    Ok(Node { kind: Kind::Bool(s == "true"), pos })
                }
                "and" | "or" | "not" | "in" => Err(self.err(format!("expected a value, found the keyword '{s}'"))),
                _ => {
                    self.i += 1;
                    Ok(Node { kind: Kind::Var(s), pos })
                }
            },
            Tok::Op("(") => {
                self.i += 1;
                let inner = self.or()?;
                self.expect_op(")", "')'")?;
                Ok(inner)
            }
            t => Err(self.err(format!("expected a value, found {}", describe(&t)))),
        }
    }
}

// ------------------------------------------------------------- evaluation

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ty {
    Num,
    Bool,
}

/// Where a named value comes from: an axis of the coordinates, or an
/// attribute. Both borrow the cloud's arrays — nothing is copied to evaluate.
#[derive(Clone, Copy)]
enum Column<'a> {
    Coord(&'a [Point], usize),
    Attr(&'a Attr),
}

/// The parsed expression, ready to run against a cloud: the syntax tree with
/// every name already resolved to a column and every type already checked, so
/// evaluating a chunk of points is arithmetic and nothing else.
///
/// `Box<C>` is a child node held on the heap. A node cannot simply contain
/// another node of the same type — that would be a value of infinite size —
/// so the children sit behind a pointer, as they would in C.
enum C<'a> {
    Num(f64),
    Bool(bool),
    NumCol(Column<'a>),
    BoolCol(&'a [bool]),
    ToNum(Box<C<'a>>),
    Neg(Box<C<'a>>),
    Not(Box<C<'a>>),
    Arith(ArOp, Box<C<'a>>, Box<C<'a>>),
    Cmp(CmpOp, Box<C<'a>>, Box<C<'a>>),
    And(Box<C<'a>>, Box<C<'a>>),
    Or(Box<C<'a>>, Box<C<'a>>),
    In(Box<C<'a>>, &'a [f64], bool),
}

/// A value over a chunk of points: one scalar for all, or one per point.
///
/// Keeping the two apart is what makes `height > 2` cheap: the right-hand side
/// stays a single number instead of being copied into a full-length array.
/// `<T>` is a type parameter, so the same enum serves `f64` values and `bool`
/// ones without being written twice.
enum Val<T> {
    S(T),
    V(Vec<T>),
}

impl<T: Copy> Val<T> {
    #[inline]
    fn get(&self, i: usize) -> T {
        match self {
            Val::S(v) => *v,
            Val::V(v) => v[i],
        }
    }

    fn map<U>(self, f: impl Fn(T) -> U) -> Val<U> {
        match self {
            Val::S(v) => Val::S(f(v)),
            Val::V(v) => Val::V(v.into_iter().map(f).collect()),
        }
    }

    fn zip<U: Copy, R>(self, other: Val<U>, n: usize, f: impl Fn(T, U) -> R) -> Val<R> {
        match (&self, &other) {
            (Val::S(a), Val::S(b)) => Val::S(f(*a, *b)),
            _ => Val::V((0..n).map(|i| f(self.get(i), other.get(i))).collect()),
        }
    }
}

fn attr_f64(a: &Attr, s: usize, e: usize) -> Vec<f64> {
    match a {
        Attr::F64(v) => v[s..e].to_vec(),
        Attr::F32(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::I64(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::I32(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::U32(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::U16(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::U8(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::I8(v) => v[s..e].iter().map(|&x| x as f64).collect(),
        Attr::Bool(v) => v[s..e].iter().map(|&x| x as u8 as f64).collect(),
    }
}

impl C<'_> {
    fn num(&self, s: usize, e: usize) -> Val<f64> {
        let n = e - s;
        match self {
            C::Num(v) => Val::S(*v),
            C::NumCol(Column::Coord(p, k)) => Val::V(p[s..e].iter().map(|q| q[*k]).collect()),
            C::NumCol(Column::Attr(a)) => Val::V(attr_f64(a, s, e)),
            C::ToNum(b) => b.boolean(s, e).map(|b| b as u8 as f64),
            C::Neg(a) => a.num(s, e).map(|v| -v),
            C::Arith(op, a, b) => {
                let (a, b) = (a.num(s, e), b.num(s, e));
                match op {
                    ArOp::Add => a.zip(b, n, |x, y| x + y),
                    ArOp::Sub => a.zip(b, n, |x, y| x - y),
                    ArOp::Mul => a.zip(b, n, |x, y| x * y),
                    ArOp::Div => a.zip(b, n, |x, y| x / y),
                }
            }
            _ => unreachable!("type-checked"),
        }
    }

    fn boolean(&self, s: usize, e: usize) -> Val<bool> {
        let n = e - s;
        match self {
            C::Bool(v) => Val::S(*v),
            C::BoolCol(v) => Val::V(v[s..e].to_vec()),
            C::Not(a) => a.boolean(s, e).map(|v| !v),
            C::And(a, b) => a.boolean(s, e).zip(b.boolean(s, e), n, |x, y| x && y),
            C::Or(a, b) => a.boolean(s, e).zip(b.boolean(s, e), n, |x, y| x || y),
            C::Cmp(op, a, b) => {
                let (a, b) = (a.num(s, e), b.num(s, e));
                match op {
                    CmpOp::Lt => a.zip(b, n, |x, y| x < y),
                    CmpOp::Le => a.zip(b, n, |x, y| x <= y),
                    CmpOp::Gt => a.zip(b, n, |x, y| x > y),
                    CmpOp::Ge => a.zip(b, n, |x, y| x >= y),
                    CmpOp::Eq => a.zip(b, n, |x, y| x == y),
                    CmpOp::Ne => a.zip(b, n, |x, y| x != y),
                }
            }
            C::In(a, set, negate) => a.num(s, e).map(|v| set.contains(&v) != *negate),
            _ => unreachable!("type-checked"),
        }
    }
}

/// Edit distance, for "did you mean" hints.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for j in 0..b.len() {
            let cur = row[j + 1];
            row[j + 1] = (prev + (ca != b[j]) as usize).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

const COORDS: [&str; 3] = ["x", "y", "z"];

impl Expr {
    /// Parse `src`.
    ///
    /// # Errors
    /// A syntax error, with its character position and the expression
    /// marked at that position.
    pub fn parse(src: &str) -> Result<Self> {
        let toks = lex(src)?;
        if toks.len() == 1 {
            return Err(syntax(src, 0, "the expression is empty"));
        }
        let mut p = Parser { src, toks, i: 0 };
        let root = p.or()?;
        if p.peek().tok != Tok::End {
            let t = describe(&p.peek().tok);
            return Err(p.err(format!("{t} is not expected here; expected an operator or the end of the expression")));
        }
        Ok(Expr { src: src.to_string(), root })
    }

    /// The source text.
    pub fn source(&self) -> &str {
        &self.src
    }

    /// Attribute names the expression refers to (other than `x`, `y`, `z`),
    /// sorted and without repeats.
    pub fn names(&self) -> Vec<String> {
        fn walk(n: &Node, out: &mut Vec<String>) {
            match &n.kind {
                Kind::Var(s) if !COORDS.contains(&s.as_str()) => out.push(s.clone()),
                Kind::Neg(a) | Kind::Not(a) | Kind::In(a, _, _) => walk(a, out),
                Kind::Arith(_, a, b) | Kind::Cmp(_, a, b) | Kind::And(a, b) | Kind::Or(a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &mut out);
        out.sort();
        out.dedup();
        out
    }

    /// Check that every name the expression uses is `x`, `y`, `z` or one of
    /// `available`.
    ///
    /// # Errors
    /// The first unknown name, with its position, the names available and
    /// the closest one if it is a likely misspelling.
    pub fn check_names<S: AsRef<str>>(&self, available: &[S]) -> Result<()> {
        fn first_unknown<'n, S: AsRef<str>>(n: &'n Node, available: &[S]) -> Option<(&'n str, usize)> {
            match &n.kind {
                Kind::Var(s) if !COORDS.contains(&s.as_str()) && !available.iter().any(|a| a.as_ref() == s) => Some((s.as_str(), n.pos)),
                Kind::Neg(a) | Kind::Not(a) | Kind::In(a, _, _) => first_unknown(a, available),
                Kind::Arith(_, a, b) | Kind::Cmp(_, a, b) | Kind::And(a, b) | Kind::Or(a, b) => first_unknown(a, available).or_else(|| first_unknown(b, available)),
                _ => None,
            }
        }
        let Some((name, pos)) = first_unknown(&self.root, available) else { return Ok(()) };
        let mut all: Vec<&str> = COORDS.to_vec();
        let mut attrs: Vec<&str> = available.iter().map(|a| a.as_ref()).filter(|a| !COORDS.contains(a)).collect();
        attrs.sort();
        all.extend(attrs);
        let hint = all.iter().map(|c| (distance(name, c), *c)).filter(|(d, c)| *d <= 2 && *d < c.chars().count()).min().map(|(_, c)| format!(" (did you mean '{c}'?)")).unwrap_or_default();
        Err(Error::invalid(marked(&self.src, pos, format!("unknown attribute '{name}' at position {pos}{hint}; the cloud has: {}", all.join(", ")))))
    }

    /// Turn one node of the parsed expression into something executable, and
    /// say what type it produces.
    ///
    /// This is where names are resolved and types are checked, so that
    /// `height > 2` is rejected before a single point is touched if the cloud
    /// has no height. Each node becomes a `C`, which borrows the arrays it
    /// reads - hence the `'a` tying the result to the cloud's lifetime - so
    /// evaluating it later copies nothing. Mixing a number and a condition
    /// inserts a conversion rather than failing, which is what makes
    /// `classification == 2 + 0` behave.
    fn compile<'a>(&self, n: &Node, xyz: &'a [Point], attrs: &'a BTreeMap<String, Attr>, sets: &'a [Vec<f64>], next_set: &mut usize) -> Result<(C<'a>, Ty)> {
        let type_err = |pos: usize, msg: &str| Error::invalid(marked(&self.src, pos, format!("type error at position {pos}: {msg}")));
        let as_num = |(c, t): (C<'a>, Ty)| if t == Ty::Bool { C::ToNum(Box::new(c)) } else { c };
        Ok(match &n.kind {
            Kind::Num(v) => (C::Num(*v), Ty::Num),
            Kind::Bool(b) => (C::Bool(*b), Ty::Bool),
            Kind::Var(s) => match COORDS.iter().position(|c| c == s) {
                Some(k) => (C::NumCol(Column::Coord(xyz, k)), Ty::Num),
                None => match &attrs[s] {
                    Attr::Bool(v) => (C::BoolCol(v), Ty::Bool),
                    a => (C::NumCol(Column::Attr(a)), Ty::Num),
                },
            },
            Kind::Neg(a) => (C::Neg(Box::new(as_num(self.compile(a, xyz, attrs, sets, next_set)?))), Ty::Num),
            Kind::Arith(op, a, b) => {
                let a = as_num(self.compile(a, xyz, attrs, sets, next_set)?);
                let b = as_num(self.compile(b, xyz, attrs, sets, next_set)?);
                (C::Arith(*op, Box::new(a), Box::new(b)), Ty::Num)
            }
            Kind::Cmp(op, a, b) => {
                let a = as_num(self.compile(a, xyz, attrs, sets, next_set)?);
                let b = as_num(self.compile(b, xyz, attrs, sets, next_set)?);
                (C::Cmp(*op, Box::new(a), Box::new(b)), Ty::Bool)
            }
            Kind::In(a, _, negate) => {
                let a = as_num(self.compile(a, xyz, attrs, sets, next_set)?);
                let set = &sets[*next_set];
                *next_set += 1;
                (C::In(Box::new(a), set, *negate), Ty::Bool)
            }
            Kind::Not(a) | Kind::And(a, _) | Kind::Or(a, _) => {
                let mut side = |x: &Node| -> Result<C<'a>> {
                    let (c, t) = self.compile(x, xyz, attrs, sets, next_set)?;
                    if t == Ty::Num {
                        return Err(type_err(x.pos, "this is a number, not a condition; compare it (for example 'intensity > 0')"));
                    }
                    Ok(c)
                };
                match &n.kind {
                    Kind::Not(_) => (C::Not(Box::new(side(a)?)), Ty::Bool),
                    Kind::And(_, b) => {
                        let ca = side(a)?;
                        (C::And(Box::new(ca), Box::new(side(b)?)), Ty::Bool)
                    }
                    Kind::Or(_, b) => {
                        let ca = side(a)?;
                        (C::Or(Box::new(ca), Box::new(side(b)?)), Ty::Bool)
                    }
                    _ => unreachable!(),
                }
            }
        })
    }

    /// Evaluate the expression on every point.
    ///
    /// # Errors
    /// An unknown name (see [`Expr::check_names`]), an attribute whose length
    /// is not the number of points, or an expression that gives numbers
    /// rather than a condition.
    pub fn mask(&self, xyz: &[Point], attrs: &BTreeMap<String, Attr>) -> Result<Vec<bool>> {
        self.check_names(&attrs.keys().collect::<Vec<_>>())?;
        let n = xyz.len();
        for name in self.names() {
            let len = attrs[&name].len();
            if len != n {
                return Err(Error::invalid(format!("attribute '{name}' has {len} values for {n} points")));
            }
        }
        // 'in' lists, in the order compile() meets them.
        fn sets(node: &Node, out: &mut Vec<Vec<f64>>) {
            match &node.kind {
                Kind::In(a, s, _) => {
                    sets(a, out);
                    out.push(s.clone());
                }
                Kind::Neg(a) | Kind::Not(a) => sets(a, out),
                Kind::Arith(_, a, b) | Kind::Cmp(_, a, b) | Kind::And(a, b) | Kind::Or(a, b) => {
                    sets(a, out);
                    sets(b, out);
                }
                _ => {}
            }
        }
        let mut lists = Vec::new();
        sets(&self.root, &mut lists);
        let (c, ty) = self.compile(&self.root, xyz, attrs, &lists, &mut 0)?;
        if ty == Ty::Num {
            return Err(Error::invalid(marked(&self.src, 0, "type error: the expression gives numbers, not a condition; compare it (for example 'height > 2')")));
        }
        let mut out = vec![false; n];
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(k, o)| {
            let s = k * CHUNK;
            match c.boolean(s, s + o.len()) {
                Val::S(v) => o.fill(v),
                Val::V(v) => o.copy_from_slice(&v),
            }
        });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud() -> (Vec<Point>, BTreeMap<String, Attr>) {
        let xyz = vec![[0.0, 0.0, 0.5], [1.0, 2.0, 3.0], [2.0, 4.0, f64::NAN], [3.0, 6.0, 10.0]];
        let mut attrs = BTreeMap::new();
        attrs.insert("classification".into(), Attr::U8(vec![2, 5, 2, 7]));
        attrs.insert("height".into(), Attr::F32(vec![0.1, 2.5, 3.0, f32::NAN]));
        attrs.insert("withheld".into(), Attr::Bool(vec![false, true, false, true]));
        (xyz, attrs)
    }

    fn eval(src: &str) -> Vec<bool> {
        let (xyz, attrs) = cloud();
        Expr::parse(src).unwrap().mask(&xyz, &attrs).unwrap()
    }

    fn err(src: &str) -> String {
        let (xyz, attrs) = cloud();
        match Expr::parse(src) {
            Err(e) => e.to_string(),
            Ok(e) => e.mask(&xyz, &attrs).unwrap_err().to_string(),
        }
    }

    #[test]
    fn comparisons_and_logic() {
        assert_eq!(eval("height > 2 & classification != 2"), vec![false, true, false, false]);
        assert_eq!(eval("height > 2 and classification != 2"), vec![false, true, false, false]);
        assert_eq!(eval("height > 2 | classification == 7"), vec![false, true, true, true]);
        assert_eq!(eval("!(z < 1) && not withheld"), vec![false, false, true, false]);
        assert_eq!(eval("classification in (2, 7)"), vec![true, false, true, true]);
        assert_eq!(eval("classification not in (2, 7,)"), vec![false, true, false, false]);
        assert_eq!(eval("withheld"), vec![false, true, false, true]);
        assert_eq!(eval("withheld == 1"), vec![false, true, false, true]);
        assert_eq!(eval("true"), vec![true; 4]);
    }

    #[test]
    fn arithmetic_precedence_and_chains() {
        assert_eq!(eval("x + y * 2 == 5"), vec![false, true, false, false]);
        assert_eq!(eval("(x + y) * 2 == 6"), vec![false, true, false, false]);
        assert_eq!(eval("-x < -1.5"), vec![false, false, true, true]);
        assert_eq!(eval("y / x > 1.9"), vec![false, true, true, true]);
        assert_eq!(eval("1 <= y < 6"), vec![false, true, true, false]);
        assert_eq!(eval("x - -1 == 2"), vec![false, true, false, false]);
        assert_eq!(eval("1e1 == z"), vec![false, false, false, true]);
        assert_eq!(eval("x in (-1, 3)"), vec![false, false, false, true]);
    }

    #[test]
    fn nan_semantics() {
        assert_eq!(eval("z > 0"), vec![true, true, false, true]);
        assert_eq!(eval("z != z"), vec![false, false, true, false]);
        assert_eq!(eval("height <= 3 | height > 3"), vec![true, true, true, false]);
    }

    #[test]
    fn errors_name_the_problem_and_position() {
        let e = err("hieght > 2");
        assert!(e.contains("unknown attribute 'hieght' at position 0 (did you mean 'height'?)"), "{e}");
        assert!(e.contains("x, y, z, classification, height, withheld"), "{e}");
        let e = err("height > (2");
        assert!(e.contains("syntax error at position 11: expected ')'"), "{e}");
        assert!(e.contains(&format!("\n    height > (2\n    {}^", " ".repeat(11))), "{e}");
        let e = err("height = 2");
        assert!(e.contains("position 7: use '=='"), "{e}");
        let e = err("height > 2 2");
        assert!(e.contains("position 11: the number 2 is not expected here"), "{e}");
        let e = err("height & z > 1");
        assert!(e.contains("type error at position 0"), "{e}");
        let e = err("height + 1");
        assert!(e.contains("gives numbers, not a condition"), "{e}");
        let e = err("classification in 2");
        assert!(e.contains("position 18: expected '(' to start the list"), "{e}");
        let e = err("classification in (a)");
        assert!(e.contains("expected a number in the 'in' list"), "{e}");
        assert!(err("").contains("empty"));
        assert!(err("z > 1 &").contains("position 7: expected a value"));
        assert!(err("z $ 1").contains("unexpected character '$'"));
        assert!(err("2m > 1").contains("position 1"));
    }

    #[test]
    fn names_and_many_points() {
        let e = Expr::parse("height > 2 & classification != 2 & z > 0 & height < 9").unwrap();
        assert_eq!(e.names(), vec!["classification".to_string(), "height".to_string()]);
        let n = 3 * CHUNK + 17;
        let xyz: Vec<Point> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
        let mut attrs = BTreeMap::new();
        attrs.insert("id".into(), Attr::I64((0..n as i64).collect()));
        let m = Expr::parse("x == id & id / 2 == 90000 | id == 5").unwrap().mask(&xyz, &attrs).unwrap();
        assert_eq!(m.iter().filter(|v| **v).count(), 2);
        assert!(m[5] && m[180_000]);
        assert!(Expr::parse("z > 0").unwrap().mask(&[], &BTreeMap::new()).unwrap().is_empty());
    }
}
