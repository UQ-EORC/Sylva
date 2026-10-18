// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! `.rxp` reading the way RiSCAN PRO converts a scan: every laser shot from
//! RiVLib's C++ point-cloud class (through the `sylva_rivlib` shim, built from
//! `crates/sylva-rs/rivlib-shim/`), multiple-time-around resolution with
//! [`super::mta`], then the same deterministic filters rimta5 applies.
//!
//! What this reproduces, measured against rimta5 6.9.6 on a VZ-600i scan
//! (ScanPos001 of the Pilliga project): 97-99% identical point set (99% on
//! lines with little far canopy, 97% where zone-2 returns are 6% of the
//! points), coordinates within 0.25 mm (the rdbx rounding) for the common
//! points, zone choice 98.3-99.6%. What it does not: the dynamic window analysis
//! and the near-range window correction (only applied when that analysis
//! succeeds, which it did not on the reference scans), and rimta5's exact
//! rule for the last fraction of a percent of uncertain echoes.
//!
//! Pipeline per scan line, streamed with a window of lines around it:
//!
//! 1. echoes are attached to the pulse whose time equals theirs (late echoes
//!    arrive after the next shot) and carry RiVLib's range under each zone;
//! 2. [`mta::stage1`], [`mta::stage2`] and [`mta::stage3`] pick the zone, each
//!    pass seeing the previous one's results for the neighbouring lines;
//! 3. the echo is placed on the pulse `zone - 1` shots earlier at that range,
//!    along that pulse's beam, with that pulse's timestamp;
//! 4. filters: amplitude 0, the dynamic-trigger rule (amplitude below 2.5 dB
//!    within 18 m), the range gate, the notch window on zone-1 echoes
//!    ("notch after MTA"), MTA-uncertain unless kept on request;
//! 5. return numbers are recounted per pulse over the surviving echoes.

use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use super::mta::{self, Decision, Grid, LineEchoes, MtaParams, MAXZ};
use super::mta_model::{EchoInfo, MtaModel};
use super::riegl::{search_dir, search_roots, RxpOptions};
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::PointCloud;

/// How [`super::riegl::read_rxp`] treats multiple-time-around ambiguity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MtaMode {
    /// Use the shim when it can be found, otherwise fall back to plain
    /// `libscanifc` reading (zone 1, no resolution).
    #[default]
    Auto,
    /// Plain `libscanifc` reading.
    Off,
    /// Require the shim; fail if it is missing.
    On,
}

/// Which rule turns the per-zone evidence into a decision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MtaDecision {
    /// The built-in learned model (trained on rimta5's output of a VZ-600i scan).
    #[default]
    Default,
    /// The hand-written three-pass thresholds only.
    Rule,
    /// A model JSON produced by `train_mta.py`.
    Model(PathBuf),
}

/// Options of the MTA path beyond [`MtaParams`].
#[derive(Clone, Debug)]
pub struct MtaOptions {
    pub mode: MtaMode,
    /// How the zone and keep/drop are decided from the evidence.
    pub decision: MtaDecision,
    /// Explicit `libsylva_rivlib` file or directory.
    pub shim: Option<PathBuf>,
    /// Zone hypotheses asked from RiVLib (1..=8).
    pub max_zone: usize,
    /// Keep MTA-uncertain echoes (flagged `mta_uncertain_point`) instead of dropping them.
    pub keep_uncertain: bool,
    /// Dynamic-trigger rule: drop echoes below this amplitude (dB) ...
    pub weak_amplitude: f32,
    /// ... when their zone-1 range is below this (m).
    pub weak_range: f32,
    /// Apply the RXP's notch window to resolved zone-1 echoes.
    pub notch: bool,
    /// Stop after this many scan lines (tests, quick looks).
    pub max_lines: Option<usize>,
    pub params: MtaParams,
}

impl Default for MtaOptions {
    fn default() -> Self {
        MtaOptions {
            mode: MtaMode::Auto,
            decision: MtaDecision::Default,
            shim: None,
            max_zone: 6,
            keep_uncertain: false,
            weak_amplitude: 2.5,
            weak_range: 18.0,
            notch: true,
            max_lines: None,
            params: MtaParams::default(),
        }
    }
}

/// What happened to the echoes of the last [`read_rxp_mta`] call (for logs and tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MtaStats {
    pub echoes: u64,
    pub amplitude_zero: u64,
    pub weak: u64,
    pub uncertain: u64,
    pub range_gate: u64,
    pub notch: u64,
    pub kept: u64,
}

// ---- the shim's C interface -------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShimShot {
    time: f64,
    line: u32,
    shot: u32,
    facet: u8,
    flags: u8,
    n_echo: u8,
    pad: u8,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShimEcho {
    range: f64,
    amplitude: f32,
    reflectance: f32,
    deviation: f32,
    background: f32,
    flags: u16,
    nzone: u16,
    zone_range: [f32; 8],
    zone_reflectance: [f32; 8],
}

type Handle = *mut c_void;
type OpenFn = unsafe extern "C" fn(*const c_char, i32, i32, *mut Handle) -> i32;
type ReadFn = unsafe extern "C" fn(Handle, u32, *mut ShimShot, *mut u32, *mut ShimEcho, u32, *mut u32, *mut i32) -> i32;
type MetaFn = unsafe extern "C" fn(Handle, *mut c_char, u32) -> u32;
type ErrFn = unsafe extern "C" fn(Handle) -> *const c_char;
type CloseFn = unsafe extern "C" fn(Handle);
type VersionFn = unsafe extern "C" fn() -> i32;

const SHIM_NAMES: &[&str] = &["libsylva_rivlib.so", "sylva_rivlib.dll", "libsylva_rivlib.dylib"];
const API_VERSION: i32 = 1;

static SHIM: OnceLock<Mutex<Option<Library>>> = OnceLock::new();

/// Locate `libsylva_rivlib`: `hint`, `$SYLVA_RIVLIB_SHIM`, then the RiVLib
/// search roots (`$RIVLIB_PATH`, `~/.local/lib`, `/opt`, ...).
pub fn find_shim(hint: Option<&Path>) -> Result<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(h) = hint {
        roots.push(h.to_path_buf());
    }
    if let Ok(v) = std::env::var("SYLVA_RIVLIB_SHIM") {
        roots.push(PathBuf::from(v));
    }
    roots.extend(search_roots());
    for root in roots {
        if root.is_file() {
            return Ok(root);
        }
        if let Some(p) = search_dir(&root, 3, SHIM_NAMES) {
            return Ok(p);
        }
    }
    Err(Error::Riegl(
        "libsylva_rivlib not found: build it with crates/sylva-rs/rivlib-shim/build.sh (needs the RiVLib SDK) \
         and set SYLVA_RIVLIB_SHIM or put it next to libscanifc"
            .into(),
    ))
}

fn with_shim<T>(hint: Option<&Path>, f: impl FnOnce(&Library) -> Result<T>) -> Result<T> {
    let cell = SHIM.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if guard.is_none() {
        let path = find_shim(hint)?;
        // SAFETY: the shim is our own C library with no load-time side effects.
        let lib = unsafe { Library::new(&path) }.map_err(|e| Error::Riegl(format!("{}: {e}", path.display())))?;
        *guard = Some(lib);
    }
    f(guard.as_ref().unwrap())
}

// ---- per-line buffers -----------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ShotRec {
    time: f64,
    o: [f32; 3],
    d: [f32; 3],
    facet: u8,
}

#[derive(Clone, Copy)]
struct EchoRec {
    shot: u32,
    r1: f64,
    amp: f32,
    dev: f32,
    flags: u16,
    zr: [f32; MAXZ],
    zrefl: [f32; MAXZ],
}

struct Line {
    index: u32,
    shots: Vec<ShotRec>,
    echoes: Vec<EchoRec>,
    le: LineEchoes,
    grid: Option<Grid>,
    dec: Option<Vec<Decision>>,
    anchored: bool,
    s2_done: bool,
    emitted: bool,
}

impl Line {
    fn new(index: u32) -> Line {
        Line { index, shots: Vec::new(), echoes: Vec::new(), le: LineEchoes::default(), grid: None, dec: None, anchored: false, s2_done: false, emitted: false }
    }
    fn finish(&mut self, nzone: usize, p: &MtaParams) {
        self.le.n_shots = self.shots.len();
        self.le.nzone = nzone;
        self.le.shot = self.echoes.iter().map(|e| e.shot).collect();
        self.le.zr = self.echoes.iter().map(|e| e.zr).collect();
        self.le.reflectance = self.echoes.iter().map(|e| e.zrefl[0]).collect();
        self.grid = Some(Grid::new(&self.le, p));
    }
}

/// Output columns, grown line by line.
#[derive(Default)]
struct Out {
    xyz: Vec<[f64; 3]>,
    amplitude: Vec<f32>,
    reflectance: Vec<f32>,
    deviation: Vec<u16>,
    echo_type: Vec<u8>,
    time: Vec<f64>,
    zone: Vec<u8>,
    line: Vec<i32>,
    shot: Vec<i32>,
    facet: Vec<u8>,
    uncertain: Vec<u8>,
    range: Vec<f32>,
    pulse: Vec<u64>,
}

struct Pipeline<'a> {
    opts: &'a RxpOptions,
    m: &'a MtaOptions,
    lines: VecDeque<Line>,
    out: Out,
    n: MtaStats,
    notch: Option<(f64, f64, f32)>,
    /// `SYLVA_MTA_FEATURES=<file>`: write one [`FEATURE_LEN`]-float record per echo
    /// (training data for a learned decision rule; see docs/guide/io.md).
    features: Option<std::io::BufWriter<std::fs::File>>,
    /// Learned decision rule, if any (None = the hand-written thresholds).
    model: Option<MtaModel>,
}

/// The model shipped with Sylva (`crates/sylva-rs/src/io/mta_model.json`), if built in.
pub const DEFAULT_MODEL_JSON: Option<&str> = default_model_json();

const fn default_model_json() -> Option<&'static str> {
    #[cfg(feature = "mta-model")]
    {
        Some(include_str!("mta_model.json"))
    }
    #[cfg(not(feature = "mta-model"))]
    {
        None
    }
}

fn load_decision(d: &MtaDecision) -> Result<Option<MtaModel>> {
    match d {
        MtaDecision::Rule => Ok(None),
        MtaDecision::Model(p) => MtaModel::load(p).map(Some),
        MtaDecision::Default => match DEFAULT_MODEL_JSON {
            Some(js) => MtaModel::from_json(js).map(Some),
            None => Ok(None),
        },
    }
}

/// Floats per echo in the feature dump: line, shot, echo index, echo count, r1, amplitude,
/// deviation, late flag, zone chosen, stage, then per zone (MAXZ): range, reflectance,
/// votes_own, votes_nb, anchors, density.
pub const FEATURE_LEN: usize = 10 + 6 * MAXZ;

impl Pipeline<'_> {
    /// Pass 1 for every line whose neighbours exist (or all, when `force`).
    fn run_stage1(&mut self, force: bool) {
        let p = &self.m.params;
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].dec.is_some() {
                continue;
            }
            if !force && i + p.line_nb >= n {
                break;
            }
            let dec = {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            self.lines[j as usize].grid.as_ref()
                        }
                    })
                    .collect();
                mta::stage1(&grids, &self.lines[i].le, p)
            };
            let line = &mut self.lines[i];
            if let Some(g) = line.grid.as_mut() {
                g.set_anchors(&line.le, &dec, p);
            }
            line.dec = Some(dec);
            line.anchored = true;
        }
    }

    /// Pass 2 for every line whose pass-1 neighbours exist; refreshes its anchors.
    fn run_stage2(&mut self, force: bool) {
        let p = self.m.params.clone();
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].s2_done || self.lines[i].dec.is_none() {
                continue;
            }
            let ready = force || (i + p.line_nb2 < n && self.lines[i + p.line_nb2].anchored);
            if !ready {
                break;
            }
            let mut dec = self.lines[i].dec.take().unwrap();
            {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb2)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb2 as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            let l = &self.lines[j as usize];
                            l.grid.as_ref().filter(|_| l.anchored)
                        }
                    })
                    .collect();
                mta::stage2(&grids, &self.lines[i].le, &mut dec, &p);
            }
            let line = &mut self.lines[i];
            if let Some(g) = line.grid.as_mut() {
                g.set_anchors(&line.le, &dec, &p);
            }
            line.dec = Some(dec);
            line.s2_done = true;
        }
    }

    /// Pass 3 and output for every line whose pass-2 neighbours exist.
    fn run_stage3(&mut self, force: bool) {
        let p = self.m.params.clone();
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].emitted || !self.lines[i].s2_done {
                continue;
            }
            let ready = force || (i + p.line_nb3 < n && self.lines[i + p.line_nb3].s2_done);
            if !ready {
                break;
            }
            let mut dec = self.lines[i].dec.take().unwrap();
            {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb3)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb3 as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            let l = &self.lines[j as usize];
                            l.grid.as_ref().filter(|_| l.s2_done)
                        }
                    })
                    .collect();
                mta::stage3(&grids, &self.lines[i].le, &mut dec, &p);
            }
            self.emit(i, &dec);
            self.lines[i].dec = Some(dec);
            self.lines[i].emitted = true;
        }
        // Drop lines no longer needed: a line stays while a later line within
        // line_nb3 is still to be emitted, plus one before it (pulse lookup).
        loop {
            let first_open = self.lines.iter().position(|l| !l.emitted).unwrap_or(self.lines.len());
            if first_open > p.line_nb3 + 1 && self.lines.len() > p.line_nb3 + 1 {
                self.lines.pop_front();
            } else {
                break;
            }
        }
    }

    fn advance(&mut self, force: bool) {
        self.run_stage1(force);
        self.run_stage2(force);
        self.run_stage3(force);
    }

    fn emit(&mut self, i: usize, dec: &[Decision]) {
        let weak_a = self.m.weak_amplitude;
        let weak_r = self.m.weak_range as f64;
        let min_r = self.opts.min_range;
        let max_r = self.opts.max_range;
        let keep_unc = self.m.keep_uncertain;
        let notch = if self.m.notch { self.notch } else { None };
        let line_idx = self.lines[i].index;
        let prev_len = if i > 0 { self.lines[i - 1].shots.len() } else { 0 };
        // the learned rule, where one is loaded, overrides zone and keep/drop
        let dec_owned: Vec<Decision>;
        let dec: &[Decision] = if let Some(model) = self.model.as_ref() {
            use rayon::prelude::*;
            let echoes = &self.lines[i].echoes;
            // pulses = runs of equal shot index; evaluate them in parallel
            let mut starts: Vec<usize> = vec![0];
            for e_i in 1..echoes.len() {
                if echoes[e_i].shot != echoes[e_i - 1].shot {
                    starts.push(e_i);
                }
            }
            if echoes.is_empty() {
                starts.clear();
            }
            starts.push(echoes.len());
            let mut v = dec.to_vec();
            let n_shots = self.lines[i].shots.len().max(1) as f32;
            let out: Vec<(usize, Decision)> = starts
                .par_windows(2)
                .flat_map_iter(|w| {
                    let (j, e2) = (w[0], w[1]);
                    let n_echoes = (e2 - j).min(255) as u8;
                    (j..e2).map(move |e_i| {
                        let e = &echoes[e_i];
                        // closer echoes of the same pulse: strongest amplitude and the gap to the nearest
                        let (mut prev_amp, mut prev_gap, mut next_amp, mut nearest) = (-1.0f32, -1.0f32, -1.0f32, f64::INFINITY);
                        for o in &echoes[j..e2] {
                            if o.r1 < e.r1 {
                                prev_amp = prev_amp.max(o.amp);
                                let gap = e.r1 - o.r1;
                                if gap < nearest {
                                    nearest = gap;
                                    prev_gap = gap as f32;
                                }
                            } else if o.r1 > e.r1 {
                                next_amp = next_amp.max(o.amp);
                            }
                        }
                        let info = EchoInfo {
                            amplitude: e.amp,
                            deviation: e.dev,
                            r1: e.r1 as f32,
                            late: e.flags & 32 != 0,
                            n_echoes,
                            idx_in_shot: (e_i - j + 1).min(255) as u8,
                            prev_amp,
                            prev_gap,
                            next_amp,
                            shot_frac: e.shot as f32 / n_shots,
                        };
                        let mut d = dec[e_i];
                        match model.decide(&e.zr, &e.zrefl, &d.evidence, info) {
                            Some((zone, p)) => {
                                d.zone = zone;
                                d.stage = if p >= model.threshold() { 4 } else { 0 };
                                d.score = p;
                            }
                            None => {
                                d.zone = 0;
                                d.stage = 0;
                            }
                        }
                        (e_i, d)
                    })
                })
                .collect();
            for (e_i, d) in out {
                v[e_i] = d;
            }
            dec_owned = v;
            &dec_owned
        } else {
            dec
        };
        if let Some(w) = self.features.as_mut() {
            use std::io::Write;
            let mut rec = [0f32; FEATURE_LEN];
            let mut buf: Vec<u8> = Vec::with_capacity(FEATURE_LEN * 4 * self.lines[i].echoes.len());
            let mut last_shot = u32::MAX;
            let mut idx_in_shot = 0u32;
            for (e_i, e) in self.lines[i].echoes.iter().enumerate() {
                if e.shot != last_shot {
                    last_shot = e.shot;
                    idx_in_shot = 0;
                }
                idx_in_shot += 1;
                let d = dec[e_i];
                rec[0] = line_idx as f32;
                rec[1] = e.shot as f32;
                rec[2] = idx_in_shot as f32;
                rec[3] = 0.0;
                rec[4] = e.r1 as f32;
                rec[5] = e.amp;
                rec[6] = e.dev;
                rec[7] = ((e.flags >> 5) & 1) as f32;
                rec[8] = d.zone as f32;
                rec[9] = d.stage as f32;
                for z in 0..MAXZ {
                    rec[10 + z] = e.zr[z];
                    rec[10 + MAXZ + z] = e.zrefl[z];
                    rec[10 + 2 * MAXZ + z] = d.evidence.votes_own[z];
                    rec[10 + 3 * MAXZ + z] = d.evidence.votes_nb[z];
                    rec[10 + 4 * MAXZ + z] = d.evidence.anchors[z];
                    rec[10 + 5 * MAXZ + z] = d.evidence.density[z];
                }
                for v in rec {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            let _ = w.write_all(&buf);
        }
        for (e_i, e) in self.lines[i].echoes.iter().enumerate() {
            self.n.echoes += 1;
            if e.amp <= 0.0 {
                self.n.amplitude_zero += 1;
                continue;
            }
            if e.amp < weak_a && e.r1 < weak_r {
                self.n.weak += 1;
                continue;
            }
            let d = dec[e_i];
            if d.zone == 0 || (d.stage == 0 && !keep_unc) {
                self.n.uncertain += 1;
                continue;
            }
            let k = d.zone as usize;
            let r = e.zr[k - 1] as f64;
            if !r.is_finite() {
                self.n.uncertain += 1;
                continue;
            }
            if r < min_r || r > max_r {
                self.n.range_gate += 1;
                continue;
            }
            if k == 1 {
                if let Some((nmin, nmax, namp)) = notch {
                    if r >= nmin && r <= nmax && e.amp < namp {
                        self.n.notch += 1;
                        continue;
                    }
                }
            }
            // the pulse the echo belongs to: zone-1 shots earlier, possibly in the previous line
            let s = e.shot as i64 - (k as i64 - 1);
            let (pl, ps, shot) = if s >= 0 {
                (line_idx, s, self.lines[i].shots[s as usize])
            } else if i > 0 && prev_len as i64 + s >= 0 {
                let ps = prev_len as i64 + s;
                (self.lines[i - 1].index, ps, self.lines[i - 1].shots[ps as usize])
            } else {
                self.n.uncertain += 1;
                continue;
            };
            let x = shot.o[0] as f64 + r * shot.d[0] as f64;
            let y = shot.o[1] as f64 + r * shot.d[1] as f64;
            let z = shot.o[2] as f64 + r * shot.d[2] as f64;
            let refl = if e.zrefl[k - 1].is_finite() { e.zrefl[k - 1] } else { e.zrefl[0] };
            let o = &mut self.out;
            o.xyz.push([x, y, z]);
            o.amplitude.push(e.amp);
            o.reflectance.push(refl);
            o.deviation.push(e.dev.clamp(0.0, 65535.0) as u16);
            o.echo_type.push(((e.flags >> 6) & 3) as u8);
            o.time.push(shot.time);
            o.zone.push(k as u8);
            o.line.push(-(pl as i32) - 1); // RiSCAN's convention: negative, -1 is the first line
            o.shot.push(ps as i32);
            o.facet.push(shot.facet);
            o.uncertain.push((d.stage == 0) as u8);
            o.range.push(r as f32);
            o.pulse.push(((pl as u64) << 24) | ps as u64);
            self.n.kept += 1;
        }
    }
}

fn json_f64(meta: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = meta.find(&pat)? + pat.len();
    let rest = &meta[i..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Read an `.rxp` with MTA resolution through the shim.
pub fn read_rxp_mta(path: &Path, opts: &RxpOptions, m: &MtaOptions) -> Result<PointCloud> {
    read_rxp_mta_with_stats(path, opts, m).map(|(c, _)| c)
}

/// [`read_rxp_mta`] that also reports what happened to the echoes.
pub fn read_rxp_mta_with_stats(path: &Path, opts: &RxpOptions, m: &MtaOptions) -> Result<(PointCloud, MtaStats)> {
    with_shim(m.shim.as_deref(), |lib| {
        // SAFETY: symbol signatures follow sylva_rivlib.cpp.
        let (open, read, meta, last_error, close, version) = unsafe {
            let open: Symbol<OpenFn> = lib.get(b"sylva_rxp_open\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let read: Symbol<ReadFn> = lib.get(b"sylva_rxp_read\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let meta: Symbol<MetaFn> = lib.get(b"sylva_rxp_meta\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let last_error: Symbol<ErrFn> = lib.get(b"sylva_rxp_last_error\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let close: Symbol<CloseFn> = lib.get(b"sylva_rxp_close\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let version: Symbol<VersionFn> = lib.get(b"sylva_rxp_api_version\0").map_err(|e| Error::Riegl(e.to_string()))?;
            (open, read, meta, last_error, close, version)
        };
        // SAFETY: plain call.
        let v = unsafe { version() };
        if v != API_VERSION {
            return Err(Error::Riegl(format!("libsylva_rivlib API version {v}, Sylva expects {API_VERSION}: rebuild the shim")));
        }
        let uri = CString::new(format!("file:{}", path.canonicalize().unwrap_or(path.to_path_buf()).display())).map_err(|e| Error::Riegl(e.to_string()))?;
        let mut handle: Handle = std::ptr::null_mut();
        let max_zone = m.max_zone.clamp(1, MAXZ) as i32;
        // SAFETY: valid C string and out-pointer; the notch is applied by us afterwards.
        let rc = unsafe { open(uri.as_ptr(), max_zone, 0, &mut handle) };
        let describe = |what: &str| -> Error {
            // SAFETY: handle is valid until close.
            let msg = unsafe { CStr::from_ptr(last_error(handle)) }.to_string_lossy().into_owned();
            Error::Riegl(format!("{}: {what}: {msg}", path.display()))
        };
        if rc != 0 {
            let e = describe("cannot open");
            // SAFETY: handle was allocated by open even on failure.
            unsafe { close(handle) };
            return Err(e);
        }
        const SHOTS: usize = 4096;
        const ECHOES: usize = SHOTS * 8 + 256;
        let zero_shot = ShimShot { time: 0.0, line: 0, shot: 0, facet: 0, flags: 0, n_echo: 0, pad: 0, ox: 0.0, oy: 0.0, oz: 0.0, dx: 0.0, dy: 0.0, dz: 0.0 };
        let zero_echo = ShimEcho { range: 0.0, amplitude: 0.0, reflectance: 0.0, deviation: 0.0, background: 0.0, flags: 0, nzone: 0, zone_range: [0.0; 8], zone_reflectance: [0.0; 8] };
        let mut shot_buf = vec![zero_shot; SHOTS];
        let mut echo_buf = vec![zero_echo; ECHOES];
        let features = std::env::var_os("SYLVA_MTA_FEATURES").and_then(|f| std::fs::File::create(f).ok()).map(std::io::BufWriter::new);
        let model = load_decision(&m.decision)?;
        let mut pipe = Pipeline { opts, m, lines: VecDeque::new(), out: Out::default(), n: MtaStats::default(), notch: None, features, model };
        let mut cur: Option<Line> = None;
        let mut meta_read = false;
        let nzone = max_zone as usize;
        let mut stop = false;
        loop {
            let (mut ns, mut ne, mut eof) = (0u32, 0u32, 0i32);
            // SAFETY: buffers sized as declared.
            let rc = unsafe { read(handle, SHOTS as u32, shot_buf.as_mut_ptr(), &mut ns, echo_buf.as_mut_ptr(), ECHOES as u32, &mut ne, &mut eof) };
            if rc != 0 {
                let e = describe("read error");
                // SAFETY: handle valid.
                unsafe { close(handle) };
                return Err(e);
            }
            if !meta_read && ns > 0 {
                meta_read = true;
                let mut buf = vec![0u8; 1 << 16];
                // SAFETY: buffer size passed.
                let need = unsafe { meta(handle, buf.as_mut_ptr() as *mut c_char, buf.len() as u32) } as usize;
                if need > buf.len() {
                    buf = vec![0u8; need];
                    // SAFETY: as above.
                    unsafe { meta(handle, buf.as_mut_ptr() as *mut c_char, buf.len() as u32) };
                }
                let js = String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string();
                if let (Some(a), Some(b), Some(c)) = (json_f64(&js, "range_min"), json_f64(&js, "range_max"), json_f64(&js, "amplitude_max")) {
                    pipe.notch = Some((a, b, c as f32));
                }
            }
            let mut ei = 0usize;
            for si in 0..ns as usize {
                let s = shot_buf[si];
                let (line_no, shot_no, n_echo) = (s.line, s.shot, s.n_echo as usize);
                if cur.as_ref().map(|l| l.index != line_no).unwrap_or(true) {
                    if let Some(mut l) = cur.take() {
                        l.finish(nzone, &m.params);
                        pipe.lines.push_back(l);
                        pipe.advance(false);
                        if let Some(max) = m.max_lines {
                            if line_no as usize >= max {
                                stop = true;
                            }
                        }
                    }
                    if stop {
                        break;
                    }
                    cur = Some(Line::new(line_no));
                }
                let l = cur.as_mut().unwrap();
                debug_assert_eq!(shot_no as usize, l.shots.len());
                let (time, ox, oy, oz, dx, dy, dz, facet) = (s.time, s.ox, s.oy, s.oz, s.dx, s.dy, s.dz, s.facet);
                l.shots.push(ShotRec { time, o: [ox, oy, oz], d: [dx, dy, dz], facet });
                for _ in 0..n_echo {
                    let e = echo_buf[ei];
                    ei += 1;
                    let (r1, amp, dev, flags, zr, zrefl) = (e.range, e.amplitude, e.deviation, e.flags, e.zone_range, e.zone_reflectance);
                    if opts.drop_pseudo_echoes && flags & super::riegl::FLAG_PSEUDO_ECHO != 0 {
                        continue;
                    }
                    l.echoes.push(EchoRec { shot: shot_no, r1, amp, dev, flags, zr, zrefl });
                }
            }
            if stop || eof != 0 {
                break;
            }
        }
        // SAFETY: handle valid.
        unsafe { close(handle) };
        if !stop {
            if let Some(mut l) = cur.take() {
                l.finish(nzone, &m.params);
                pipe.lines.push_back(l);
            }
        }
        pipe.advance(true);
        let stats = pipe.n;
        Ok((finish(pipe.out, opts, m), stats))
    })
}

/// Return numbers, echo/stride selection and the final point cloud.
fn finish(o: Out, opts: &RxpOptions, m: &MtaOptions) -> PointCloud {
    let np = o.xyz.len();
    // return_number / number_of_returns per pulse, by range
    let mut order: Vec<u32> = (0..np as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        o.pulse[a as usize].cmp(&o.pulse[b as usize]).then(o.range[a as usize].partial_cmp(&o.range[b as usize]).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut ret = vec![0u8; np];
    let mut cnt = vec![0u8; np];
    let mut pulse_ord = vec![0usize; np];
    let mut i = 0;
    let mut n_pulses = 0usize;
    while i < np {
        let mut j = i;
        while j < np && o.pulse[order[j] as usize] == o.pulse[order[i] as usize] {
            j += 1;
        }
        let c = (j - i).min(255) as u8;
        for (r, &ix) in order[i..j].iter().enumerate() {
            ret[ix as usize] = (r + 1).min(255) as u8;
            cnt[ix as usize] = c;
            pulse_ord[ix as usize] = n_pulses;
        }
        n_pulses += 1;
        i = j;
    }
    // selection (echo type, strides, cap)
    let mut keep = vec![true; np];
    let mut total = 0usize;
    for ix in 0..np {
        let k = match opts.echoes.as_str() {
            "first" => ret[ix] == 1,
            "last" => ret[ix] == cnt[ix],
            "single" => cnt[ix] == 1,
            _ => true,
        } && (opts.shot_stride <= 1 || pulse_ord[ix] % opts.shot_stride == 0)
            && (opts.stride <= 1 || (ix + 1) % opts.stride == 0)
            && opts.max_points.map(|mx| total < mx).unwrap_or(true);
        keep[ix] = k;
        if k {
            total += 1;
        }
    }
    fn sel<T: Copy>(v: &[T], keep: &[bool]) -> Vec<T> {
        v.iter().zip(keep).filter(|(_, &k)| k).map(|(x, _)| *x).collect()
    }
    let mut cloud = PointCloud::new(sel(&o.xyz, &keep));
    let a = &mut cloud.attrs;
    a.insert("amplitude".into(), Attr::F32(sel(&o.amplitude, &keep)));
    a.insert("reflectance".into(), Attr::F32(sel(&o.reflectance, &keep)));
    a.insert("deviation".into(), Attr::U16(sel(&o.deviation, &keep)));
    a.insert("echo_type".into(), Attr::U8(sel(&o.echo_type, &keep)));
    a.insert("gps_time".into(), Attr::F64(sel(&o.time, &keep)));
    a.insert("mta_zone".into(), Attr::U8(sel(&o.zone, &keep)));
    a.insert("return_number".into(), Attr::U8(sel(&ret, &keep)));
    a.insert("number_of_returns".into(), Attr::U8(sel(&cnt, &keep)));
    a.insert("scan_line_index".into(), Attr::I32(sel(&o.line, &keep)));
    a.insert("shot_index_line".into(), Attr::I32(sel(&o.shot, &keep)));
    a.insert("mirror_facet".into(), Attr::U8(sel(&o.facet, &keep)));
    if m.keep_uncertain {
        a.insert("mta_uncertain_point".into(), Attr::U8(sel(&o.uncertain, &keep)));
    }
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_shim() {
        assert_eq!(std::mem::size_of::<ShimShot>(), 44);
        assert_eq!(std::mem::size_of::<ShimEcho>(), 92);
    }

    #[test]
    fn json_numbers() {
        let js = r#"{"notch":{"range_min":-0.035,"range_max":0.244,"amplitude_max":3},"n_shots":5}"#;
        assert_eq!(json_f64(js, "range_max"), Some(0.244));
        assert_eq!(json_f64(js, "amplitude_max"), Some(3.0));
        assert_eq!(json_f64(js, "missing"), None);
    }

    #[test]
    fn missing_shim_is_a_clear_error() {
        // a hint that does not exist falls through to the usual roots; only the
        // message is checked, since a developer machine may have the shim
        if let Err(Error::Riegl(msg)) = find_shim(Some(Path::new("/definitely/not/here"))) {
            assert!(msg.contains("libsylva_rivlib"));
        }
    }
}
