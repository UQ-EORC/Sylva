// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! PulseWaves 0.3 (Isenburg 2012; the `.pls` pulse file and the `.wvs`
//! waves file of the open PulseWaves specification and its reference
//! library), uncompressed.
//!
//! A `.pls` file has a 352-byte header, VLRs of 96-byte headers, and
//! fixed-size pulse records (format 0, 48 bytes): time `T`, the byte offset
//! of the pulse's waves in the `.wvs` file, an `anchor` and a `target`
//! point (quantised like LAS coordinates), the first and last returning
//! sample, a descriptor index and flags, intensity and classification. The
//! beam passes from the anchor towards the target, and the target lies 1000
//! sampling units (of the descriptor's composition) after the anchor: sample
//! `i` of a segment whose duration from the anchor is `d` lies at
//! `anchor + (target - anchor) (d + i) / 1000`.
//!
//! A descriptor (VLR `PulseWaves_Spec`, record id 200000 + index) has a
//! composition (optical centre to anchor offset, number of samplings,
//! sampling unit in ns) and one sampling record per sampling: outgoing or
//! returning, channel, how many bits store the number of segments, the
//! duration from the anchor of each segment and its number of samples, the
//! bits per sample and an optional lookup table (VLR record id
//! 300000 + index) that turns sample values into physical units. The waves
//! of a pulse are, for each sampling and each of its segments, the
//! segment's duration and sample count where they are stored, then the
//! samples.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian as LE, ReadBytesExt, WriteBytesExt};

use super::Waveforms;
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::transform::{add, dot, norm, scale, sub};
use crate::Point;

const HEADER: usize = 352;
const VLR_HEADER: usize = 96;
const WAVES_HEADER: usize = 60;
const PULSE0: usize = 48;
const DESCRIPTOR_ID: u32 = 200_000;
const TABLE_ID: u32 = 300_000;
/// `optical_center_to_anchor_point` value meaning the offset varies.
const FLUCTUATE: i32 = 0x8FFF_FFFFu32 as i32;

/// One sampling of a composition.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    /// 0 undefined, 1 outgoing, 2 returning.
    pub kind: u8,
    pub channel: u8,
    pub bits_for_duration: u8,
    pub duration_scale: f32,
    pub duration_offset: f32,
    pub bits_for_segments: u8,
    pub bits_for_samples: u8,
    pub n_segments: u16,
    pub n_samples: u32,
    pub bits_per_sample: u16,
    pub lookup_table: u16,
    /// Sampling unit (ns).
    pub units: f32,
}

/// A pulse descriptor: composition and samplings.
#[derive(Debug, Clone, PartialEq)]
pub struct Descriptor {
    pub optical_center_to_anchor: i32,
    pub extra_waves_bytes: u16,
    pub scanner: u32,
    /// Sampling unit of the composition (ns).
    pub units: f32,
    pub samplings: Vec<Sampling>,
}

/// What a `.pls` header says.
#[derive(Debug, Clone)]
pub struct PlsInfo {
    pub version: (u8, u8),
    pub n_pulses: u64,
    pub pulse_size: u32,
    pub pulse_format: u32,
    pub pulse_compression: u32,
    pub offset_to_pulses: u64,
    pub t_scale: f64,
    pub t_offset: f64,
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    /// `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub bounds: [f64; 6],
    pub system: String,
    pub software: String,
    pub descriptors: BTreeMap<u32, Descriptor>,
    /// First lookup table of each table VLR, by index.
    pub tables: BTreeMap<u32, Vec<f32>>,
    /// The `.wvs` file, if present.
    pub waves: Option<PathBuf>,
}

fn file_err(path: &Path, msg: impl Into<String>) -> Error {
    Error::file(path, msg)
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).trim_end_matches('\0').trim().to_string()
}

fn parse_descriptor(d: &[u8]) -> Option<Descriptor> {
    if d.len() < 28 {
        return None;
    }
    let size = LE::read_u32(d) as usize;
    let n_samplings = LE::read_u16(&d[14..]) as usize;
    let mut desc = Descriptor {
        optical_center_to_anchor: LE::read_i32(&d[8..]),
        extra_waves_bytes: LE::read_u16(&d[12..]),
        units: LE::read_f32(&d[16..]),
        scanner: LE::read_u32(&d[24..]),
        samplings: Vec::with_capacity(n_samplings),
    };
    let mut p = size.max(28);
    for _ in 0..n_samplings {
        if p + 40 > d.len() {
            return None;
        }
        let s = &d[p..];
        desc.samplings.push(Sampling {
            kind: s[8],
            channel: s[9],
            bits_for_duration: s[11],
            duration_scale: LE::read_f32(&s[12..]),
            duration_offset: LE::read_f32(&s[16..]),
            bits_for_segments: s[20],
            bits_for_samples: s[21],
            n_segments: LE::read_u16(&s[22..]),
            n_samples: LE::read_u32(&s[24..]),
            bits_per_sample: LE::read_u16(&s[28..]),
            lookup_table: LE::read_u16(&s[30..]),
            units: LE::read_f32(&s[32..]),
        });
        p += (LE::read_u32(s) as usize).max(40);
    }
    Some(desc)
}

fn parse_table(d: &[u8]) -> Option<Vec<f32>> {
    if d.len() < 12 {
        return None;
    }
    let size = LE::read_u32(d) as usize;
    if LE::read_u32(&d[8..]) == 0 {
        return None;
    }
    let t = d.get(size..)?;
    if t.len() < 16 {
        return None;
    }
    let tsize = LE::read_u32(t) as usize;
    let n = LE::read_u32(&t[8..]) as usize;
    let data_type = t[14];
    if data_type != 8 {
        return None;
    }
    let e = t.get(tsize..tsize + 4 * n)?;
    Some(e.as_chunks::<4>().0.iter().map(|c| LE::read_f32(c)).collect())
}

fn waves_path(path: &Path) -> Option<PathBuf> {
    for ext in ["wvs", "WVS"] {
        let p = path.with_extension(ext);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Read the header, descriptors and lookup tables of a `.pls` file.
pub fn read_info(path: impl AsRef<Path>) -> Result<PlsInfo> {
    let path = path.as_ref();
    let mut f = BufReader::new(File::open(path).map_err(|e| file_err(path, e.to_string()))?);
    let mut h = [0u8; HEADER];
    f.read_exact(&mut h).map_err(|_| file_err(path, "not a PulseWaves file (header too short)"))?;
    if &h[0..15] != b"PulseWavesPulse" {
        return Err(file_err(path, "not a PulseWaves file (no PulseWavesPulse signature)"));
    }
    let rd = |o: usize| LE::read_f64(&h[o..]);
    let header_size = LE::read_u16(&h[174..]) as u64;
    let offset_to_pulses = LE::read_i64(&h[176..]).max(0) as u64;
    let n_pulses = LE::read_i64(&h[184..]).max(0) as u64;
    let n_vlrs = LE::read_u32(&h[216..]);
    let mut info = PlsInfo {
        version: (h[172], h[173]),
        n_pulses,
        pulse_format: LE::read_u32(&h[192..]),
        pulse_size: LE::read_u32(&h[200..]),
        pulse_compression: LE::read_u32(&h[204..]),
        offset_to_pulses,
        t_scale: rd(224),
        t_offset: rd(232),
        scale: [rd(256), rd(264), rd(272)],
        offset: [rd(280), rd(288), rd(296)],
        bounds: [rd(304), rd(320), rd(336), rd(312), rd(328), rd(344)],
        system: text(&h[40..104]),
        software: text(&h[104..168]),
        descriptors: BTreeMap::new(),
        tables: BTreeMap::new(),
        waves: waves_path(path),
    };
    f.seek(SeekFrom::Start(header_size))?;
    let mut pos = header_size;
    for _ in 0..n_vlrs {
        if pos + VLR_HEADER as u64 > offset_to_pulses {
            break;
        }
        let mut vh = [0u8; VLR_HEADER];
        f.read_exact(&mut vh).map_err(|e| file_err(path, format!("truncated VLR: {e}")))?;
        let user = text(&vh[0..16]);
        let id = LE::read_u32(&vh[16..]);
        let len = LE::read_i64(&vh[24..]).max(0) as usize;
        let mut data = vec![0u8; len];
        f.read_exact(&mut data).map_err(|e| file_err(path, format!("truncated VLR: {e}")))?;
        pos += (VLR_HEADER + len) as u64;
        if user != "PulseWaves_Spec" {
            continue;
        }
        if id > DESCRIPTOR_ID && id <= DESCRIPTOR_ID + 255 {
            if let Some(d) = parse_descriptor(&data) {
                info.descriptors.insert(id - DESCRIPTOR_ID, d);
            }
        } else if id > TABLE_ID && id <= TABLE_ID + 255 {
            if let Some(t) = parse_table(&data) {
                info.tables.insert(id - TABLE_ID, t);
            }
        }
    }
    Ok(info)
}

/// Which waveforms to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kinds {
    Returning,
    Outgoing,
    All,
}

/// Options for [`read_waveforms`].
#[derive(Debug, Clone)]
pub struct PlsReadOptions {
    pub start: u64,
    pub count: Option<u64>,
    pub kinds: Kinds,
    /// Turn sample values into physical units with the lookup tables.
    pub lookup: bool,
}

impl Default for PlsReadOptions {
    fn default() -> Self {
        PlsReadOptions { start: 0, count: None, kinds: Kinds::Returning, lookup: true }
    }
}

struct PulseRec {
    t: i64,
    offset: i64,
    anchor: Point,
    target: Point,
    descriptor: u8,
    intensity: u8,
    classification: u8,
}

/// One segment of a pulse's waves.
struct Segment {
    sampling: usize,
    segment: usize,
    duration: f64,
    samples: Vec<f32>,
}

fn read_int(r: &mut impl Read, bits: u8, signed: bool) -> std::io::Result<i64> {
    Ok(match (bits, signed) {
        (8, false) => r.read_u8()? as i64,
        (8, true) => r.read_i8()? as i64,
        (16, false) => r.read_u16::<LE>()? as i64,
        (16, true) => r.read_i16::<LE>()? as i64,
        (32, false) => r.read_u32::<LE>()? as i64,
        (32, true) => r.read_i32::<LE>()? as i64,
        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unsupported field width of {bits} bits"))),
    })
}

fn read_waves(r: &mut impl Read, desc: &Descriptor, tables: &BTreeMap<u32, Vec<f32>>, lookup: bool) -> std::io::Result<Vec<Segment>> {
    let mut extra = vec![0u8; desc.extra_waves_bytes as usize];
    r.read_exact(&mut extra)?;
    let mut out = Vec::new();
    for (m, s) in desc.samplings.iter().enumerate() {
        let n_seg = if s.bits_for_segments != 0 { read_int(r, s.bits_for_segments, false)? as usize } else { s.n_segments as usize };
        let table = if lookup && s.lookup_table != 0 { tables.get(&(s.lookup_table as u32)) } else { None };
        for k in 0..n_seg {
            let q = if s.bits_for_duration != 0 { read_int(r, s.bits_for_duration, true)? } else { 0 };
            let duration = s.duration_scale as f64 * q as f64 + s.duration_offset as f64;
            let n = if s.bits_for_samples != 0 { read_int(r, s.bits_for_samples, false)? as usize } else { s.n_samples as usize };
            let raw: Vec<u32> = match s.bits_per_sample {
                8 => {
                    let mut b = vec![0u8; n];
                    r.read_exact(&mut b)?;
                    b.into_iter().map(|x| x as u32).collect()
                }
                16 => {
                    let mut b = vec![0u8; 2 * n];
                    r.read_exact(&mut b)?;
                    b.as_chunks::<2>().0.iter().map(|c| LE::read_u16(c) as u32).collect()
                }
                bits => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{bits} bits per sample is not supported (8 or 16)"))),
            };
            let samples = match table {
                Some(t) => raw.iter().map(|&x| t.get(x as usize).copied().unwrap_or(x as f32)).collect(),
                None => raw.iter().map(|&x| x as f32).collect(),
            };
            out.push(Segment { sampling: m, segment: k, duration, samples });
        }
    }
    Ok(out)
}

/// Read the waveforms of pulses `start .. start + count`: one row per
/// segment of each sampling of the chosen kind, `pulse` being the pulse's
/// index in the file. Returns the waveforms and the next pulse index.
pub fn read_waveforms(path: impl AsRef<Path>, opts: &PlsReadOptions) -> Result<(Waveforms, u64)> {
    let path = path.as_ref();
    let info = read_info(path)?;
    if info.pulse_compression != 0 {
        return Err(file_err(path, "compressed PulseWaves (.plz / .wvz) is not supported; decompress it with pulsezip"));
    }
    if info.pulse_format != 0 || (info.pulse_size as usize) < PULSE0 {
        return Err(file_err(path, format!("pulse format {} ({} bytes) is not supported", info.pulse_format, info.pulse_size)));
    }
    let start = opts.start.min(info.n_pulses);
    let count = opts.count.unwrap_or(info.n_pulses - start).min(info.n_pulses - start);
    let ps = info.pulse_size as usize;
    let mut f = File::open(path).map_err(|e| file_err(path, e.to_string()))?;
    f.seek(SeekFrom::Start(info.offset_to_pulses + start * ps as u64))?;
    let mut buf = vec![0u8; count as usize * ps];
    f.read_exact(&mut buf).map_err(|e| file_err(path, format!("truncated pulse records: {e}")))?;
    let pulses: Vec<PulseRec> = buf
        .chunks_exact(ps)
        .map(|p| {
            let q = |o: usize, k: usize| LE::read_i32(&p[o..]) as f64 * info.scale[k] + info.offset[k];
            PulseRec {
                t: LE::read_i64(p),
                offset: LE::read_i64(&p[8..]),
                anchor: [q(16, 0), q(20, 1), q(24, 2)],
                target: [q(28, 0), q(32, 1), q(36, 2)],
                descriptor: (LE::read_u16(&p[44..]) & 0xff) as u8,
                intensity: p[46],
                classification: p[47],
            }
        })
        .collect();

    // Waves in file order.
    let mut segs: Vec<Vec<Segment>> = (0..pulses.len()).map(|_| Vec::new()).collect();
    let wanted: Vec<usize> = (0..pulses.len()).filter(|&i| pulses[i].descriptor != 0 && pulses[i].offset >= 0 && info.descriptors.contains_key(&(pulses[i].descriptor as u32))).collect();
    if !wanted.is_empty() {
        let wp = info.waves.clone().ok_or_else(|| file_err(path, "no .wvs waves file next to the pulse file"))?;
        let wfile = File::open(&wp).map_err(|e| file_err(&wp, e.to_string()))?;
        let mut r = BufReader::with_capacity(1 << 20, wfile);
        let mut sig = [0u8; WAVES_HEADER];
        r.read_exact(&mut sig).map_err(|_| file_err(&wp, "not a PulseWaves waves file"))?;
        if &sig[0..15] != b"PulseWavesWaves" {
            return Err(file_err(&wp, "not a PulseWaves waves file (no PulseWavesWaves signature)"));
        }
        if LE::read_u32(&sig[16..]) != 0 {
            return Err(file_err(&wp, "compressed waves are not supported; decompress with pulsezip"));
        }
        let mut order = wanted;
        order.sort_by_key(|&i| pulses[i].offset);
        let mut pos = WAVES_HEADER as u64;
        for i in order {
            let at = pulses[i].offset as u64;
            if at >= pos && at - pos < (1 << 20) {
                r.seek_relative((at - pos) as i64)?;
            } else {
                r.seek(SeekFrom::Start(at))?;
            }
            let desc = &info.descriptors[&(pulses[i].descriptor as u32)];
            let mut counting = CountingReader { inner: &mut r, n: 0 };
            segs[i] = read_waves(&mut counting, desc, &info.tables, opts.lookup).map_err(|e| file_err(&wp, format!("waves of pulse {}: {e}", start + i as u64)))?;
            pos = at + counting.n;
        }
    }

    let mut wf = Waveforms::default();
    let (mut kind, mut channel, mut segment, mut sampling, mut intensity, mut class, mut descriptor) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for (i, p) in pulses.iter().enumerate() {
        let Some(desc) = info.descriptors.get(&(p.descriptor as u32)) else { continue };
        let d = sub(&p.target, &p.anchor);
        let len = norm(&d);
        if !(len > 0.0) || !(desc.units > 0.0) {
            continue;
        }
        let dir = scale(&d, 1.0 / len);
        let mpns = len / 1000.0 / desc.units as f64;
        let origin = if desc.optical_center_to_anchor == FLUCTUATE { [f64::NAN; 3] } else { add(&p.anchor, &scale(&dir, -mpns * desc.units as f64 * desc.optical_center_to_anchor as f64)) };
        for s in &segs[i] {
            let sm = &desc.samplings[s.sampling];
            let keep = match opts.kinds {
                Kinds::All => true,
                Kinds::Returning => sm.kind == 2,
                Kinds::Outgoing => sm.kind == 1,
            };
            let units = if sm.units > 0.0 { sm.units as f64 } else { desc.units as f64 };
            if !keep {
                continue;
            }
            wf.pulse.push((start + i as u64) as i64);
            wf.gps_time.push(p.t as f64 * info.t_scale + info.t_offset);
            wf.origin.push(origin);
            wf.anchor.push(p.anchor);
            wf.direction.push(dir);
            wf.offset.push(s.duration * units);
            wf.interval.push(units);
            wf.metres_per_ns.push(mpns);
            wf.sample_start.push(wf.samples.len());
            wf.sample_count.push(s.samples.len() as u32);
            wf.samples.extend_from_slice(&s.samples);
            kind.push(sm.kind);
            channel.push(sm.channel);
            segment.push(s.segment as u16);
            sampling.push(s.sampling as u8);
            intensity.push(p.intensity);
            class.push(p.classification);
            descriptor.push(p.descriptor);
        }
    }
    wf.attrs.insert("kind".into(), Attr::U8(kind));
    wf.attrs.insert("channel".into(), Attr::U8(channel));
    wf.attrs.insert("segment".into(), Attr::U16(segment));
    wf.attrs.insert("sampling".into(), Attr::U8(sampling));
    wf.attrs.insert("intensity".into(), Attr::U8(intensity));
    wf.attrs.insert("classification".into(), Attr::U8(class));
    wf.attrs.insert("descriptor".into(), Attr::U8(descriptor));
    Ok((wf, start + count))
}

struct CountingReader<'a, R: Read> {
    inner: &'a mut R,
    n: u64,
}

impl<R: Read> Read for CountingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let k = self.inner.read(buf)?;
        self.n += k as u64;
        Ok(k)
    }
}

/// Options for [`write_waveforms`].
#[derive(Debug, Clone)]
pub struct PlsWriteOptions {
    /// Coordinate resolution of anchors and targets (m).
    pub scale: f64,
}

impl Default for PlsWriteOptions {
    fn default() -> Self {
        PlsWriteOptions { scale: 1e-4 }
    }
}

fn sampling_bytes(kind: u8, units: f32) -> Vec<u8> {
    let mut s = Vec::with_capacity(104);
    s.write_u32::<LE>(104).unwrap();
    s.write_u32::<LE>(0).unwrap();
    s.extend_from_slice(&[kind, 0, 0, 32]);
    s.write_f32::<LE>(0.001).unwrap();
    s.write_f32::<LE>(0.0).unwrap();
    s.extend_from_slice(&[16, 16]);
    s.write_u16::<LE>(0).unwrap();
    s.write_u32::<LE>(0).unwrap();
    s.write_u16::<LE>(16).unwrap();
    s.write_u16::<LE>(0).unwrap();
    s.write_f32::<LE>(units).unwrap();
    s.write_u32::<LE>(0).unwrap();
    let mut d = [0u8; 64];
    let t: &[u8] = if kind == 1 { b"outgoing" } else { b"returning" };
    d[..t.len()].copy_from_slice(t);
    s.extend_from_slice(&d);
    s
}

fn descriptor_bytes(units: f32, origin_known: bool) -> Vec<u8> {
    let mut c = Vec::with_capacity(92 + 208);
    c.write_u32::<LE>(92).unwrap();
    c.write_u32::<LE>(0).unwrap();
    c.write_i32::<LE>(if origin_known { 0 } else { FLUCTUATE }).unwrap();
    c.write_u16::<LE>(0).unwrap();
    c.write_u16::<LE>(2).unwrap();
    c.write_f32::<LE>(units).unwrap();
    c.write_u32::<LE>(0).unwrap();
    c.write_u32::<LE>(0).unwrap();
    let mut d = [0u8; 64];
    d[..5].copy_from_slice(b"Sylva");
    c.extend_from_slice(&d);
    c.extend(sampling_bytes(1, units));
    c.extend(sampling_bytes(2, units));
    c
}

/// Write waveforms as PulseWaves 0.3 (`path` and the `.wvs` next to it).
///
/// Consecutive rows with the same `pulse` on the same beam (direction,
/// range rate and interval) become one pulse, with its outgoing rows
/// (`kind == 1`) and returning rows as segments of an outgoing and a
/// returning sampling; other rows become pulses of their own. Where the origin is
/// known it becomes the anchor (optical centre and anchor coincide), else
/// the first row's anchor is kept. Samples are stored as 16-bit integers
/// and must round to 0..65535; segment durations to 0.001 samples.
pub fn write_waveforms(wf: &Waveforms, path: impl AsRef<Path>, opts: &PlsWriteOptions) -> Result<()> {
    let path = path.as_ref();
    wf.validate()?;
    if !(opts.scale > 0.0) {
        return Err(Error::invalid("scale must be positive"));
    }
    for &v in &wf.samples {
        let r = (v as f64).round();
        if !(0.0..=65535.0).contains(&r) {
            return Err(Error::invalid(format!("PulseWaves stores samples as 16-bit integers; a sample of {v} does not fit (0 to 65535)")));
        }
    }
    let kinds = wf.kinds();
    let same_beam = |a: usize, b: usize| {
        let d = dot(&wf.direction[a], &wf.direction[b]) / (norm(&wf.direction[a]) * norm(&wf.direction[b]));
        d > 1.0 - 1e-12 && (wf.metres_per_ns[a] - wf.metres_per_ns[b]).abs() <= 1e-12 * wf.metres_per_ns[a] && wf.interval[a] == wf.interval[b] && wf.pulse[a] == wf.pulse[b]
    };
    // Pulses as row ranges.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < wf.len() {
        let mut j = i + 1;
        while j < wf.len() && same_beam(i, j) {
            j += 1;
        }
        groups.push((i, j));
        i = j;
    }
    // Descriptors by (interval, origin known).
    let mut desc_of: BTreeMap<(u32, bool), u32> = BTreeMap::new();
    let mut group_desc = Vec::with_capacity(groups.len());
    for &(a, _) in &groups {
        let key = ((wf.interval[a] as f32).to_bits(), wf.origin[a].iter().all(|v| v.is_finite()));
        let next = desc_of.len() as u32 + 1;
        let d = *desc_of.entry(key).or_insert(next);
        if d > 255 {
            return Err(Error::invalid("more than 255 distinct sampling intervals; PulseWaves allows 255 descriptors"));
        }
        group_desc.push(d);
    }

    // Header values.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut anchors = Vec::with_capacity(groups.len());
    for &(a, _) in &groups {
        let an = if wf.origin[a].iter().all(|v| v.is_finite()) { wf.origin[a] } else { wf.anchor[a] };
        if !an.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid("anchors must be finite"));
        }
        for k in 0..3 {
            lo[k] = lo[k].min(an[k]);
            hi[k] = hi[k].max(an[k]);
        }
        anchors.push(an);
    }
    if groups.is_empty() {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }
    let off: [f64; 3] = std::array::from_fn(|k| (lo[k] / 1000.0).floor() * 1000.0);
    let t_scale = 1e-9;
    let t_min = wf.gps_time.iter().cloned().filter(|t| t.is_finite()).fold(f64::INFINITY, f64::min);
    let t_offset = if t_min.is_finite() { t_min.floor() } else { 0.0 };
    let q = |v: f64, k: usize| -> Result<i32> {
        let x = ((v - off[k]) / opts.scale).round();
        if x.abs() > i32::MAX as f64 {
            return Err(Error::invalid(format!("coordinates span too far for scale {}", opts.scale)));
        }
        Ok(x as i32)
    };

    let mut vlrs = Vec::new();
    for (&(interval_bits, known), &idx) in &desc_of {
        let data = descriptor_bytes(f32::from_bits(interval_bits), known);
        let mut vh = [0u8; VLR_HEADER];
        vh[..15].copy_from_slice(b"PulseWaves_Spec");
        LE::write_u32(&mut vh[16..], DESCRIPTOR_ID + idx);
        LE::write_i64(&mut vh[24..], data.len() as i64);
        vh[32..42].copy_from_slice(b"descriptor");
        vlrs.extend_from_slice(&vh);
        vlrs.extend(data);
    }
    let mut h = [0u8; HEADER];
    h[..15].copy_from_slice(b"PulseWavesPulse");
    h[40..45].copy_from_slice(b"Sylva");
    h[104..129].copy_from_slice(b"Sylva PulseWaves writer  ");
    h[172] = 0;
    h[173] = 3;
    LE::write_u16(&mut h[174..], HEADER as u16);
    LE::write_i64(&mut h[176..], (HEADER + vlrs.len()) as i64);
    LE::write_i64(&mut h[184..], groups.len() as i64);
    LE::write_u32(&mut h[200..], PULSE0 as u32);
    LE::write_u32(&mut h[216..], desc_of.len() as u32);
    LE::write_f64(&mut h[224..], t_scale);
    LE::write_f64(&mut h[232..], t_offset);
    for k in 0..3 {
        LE::write_f64(&mut h[256 + 8 * k..], opts.scale);
        LE::write_f64(&mut h[280 + 8 * k..], off[k]);
        LE::write_f64(&mut h[304 + 16 * k..], lo[k]);
        LE::write_f64(&mut h[312 + 16 * k..], hi[k]);
    }

    let wpath = path.with_extension("wvs");
    let mut waves = BufWriter::with_capacity(1 << 20, File::create(&wpath).map_err(|e| file_err(&wpath, e.to_string()))?);
    let mut wh = [0u8; WAVES_HEADER];
    wh[..15].copy_from_slice(b"PulseWavesWaves");
    waves.write_all(&wh)?;
    let mut woff = WAVES_HEADER as u64;
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path).map_err(|e| file_err(path, e.to_string()))?);
    let (mut t_lo, mut t_hi) = (i64::MAX, i64::MIN);
    let intensity = wf.attrs.get("intensity").map(|a| a.to_f64());
    let class = wf.attrs.get("classification").map(|a| a.to_f64());
    out.write_all(&h)?;
    out.write_all(&vlrs)?;
    for (g, &(a, b)) in groups.iter().enumerate() {
        let an = anchors[g];
        let dir = scale(&wf.direction[a], 1.0 / norm(&wf.direction[a]));
        let units = wf.interval[a] as f32 as f64;
        let step = wf.metres_per_ns[a] * units; // metres per sampling unit
        let target = add(&an, &scale(&dir, step * 1000.0));
        // Segment durations in sampling units, from the pulse's anchor.
        let dur = |r: usize| (dot(&sub(&wf.anchor[r], &an), &dir) / wf.metres_per_ns[r] + wf.offset[r]) / units;
        let mut body = Vec::new();
        let mut first_last: Option<(f64, f64)> = None;
        for want in [1i64, 2] {
            let rows: Vec<usize> = (a..b).filter(|&r| if want == 1 { kinds[r] == 1 } else { kinds[r] != 1 }).collect();
            body.write_u16::<LE>(rows.len() as u16)?;
            for r in rows {
                let d = dur(r);
                let qd = (d * 1000.0).round();
                if qd.abs() > i32::MAX as f64 {
                    return Err(Error::invalid(format!("waveform {r} starts too far from its anchor for PulseWaves")));
                }
                body.write_i32::<LE>(qd as i32)?;
                let n = wf.sample_count[r];
                if n > u16::MAX as u32 {
                    return Err(Error::invalid(format!("waveform {r} has {n} samples; PulseWaves segments hold at most 65535")));
                }
                body.write_u16::<LE>(n as u16)?;
                for &v in wf.samples_of(r) {
                    body.write_u16::<LE>((v as f64).round() as u16)?;
                }
                if want == 2 && first_last.is_none() {
                    first_last = Some((d, d + n.saturating_sub(1) as f64));
                }
            }
        }
        waves.write_all(&body)?;
        let t = ((wf.gps_time[a] - t_offset) / t_scale).round() as i64;
        t_lo = t_lo.min(t);
        t_hi = t_hi.max(t);
        let mut p = [0u8; PULSE0];
        LE::write_i64(&mut p[0..], t);
        LE::write_i64(&mut p[8..], woff as i64);
        for k in 0..3 {
            LE::write_i32(&mut p[16 + 4 * k..], q(an[k], k)?);
            LE::write_i32(&mut p[28 + 4 * k..], q(target[k], k)?);
        }
        let (fs, ls) = first_last.unwrap_or((0.0, 0.0));
        LE::write_i16(&mut p[40..], fs.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
        LE::write_i16(&mut p[42..], ls.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
        LE::write_u16(&mut p[44..], group_desc[g] as u16);
        p[46] = intensity.as_ref().map(|v| v[a].clamp(0.0, 255.0) as u8).unwrap_or(0);
        p[47] = class.as_ref().map(|v| v[a].clamp(0.0, 255.0) as u8).unwrap_or(0);
        out.write_all(&p)?;
        woff += body.len() as u64;
    }
    out.flush()?;
    waves.flush()?;
    drop(out);
    // Time range.
    if !groups.is_empty() {
        let mut f = std::fs::OpenOptions::new().write(true).open(path)?;
        f.seek(SeekFrom::Start(240))?;
        f.write_i64::<LE>(t_lo)?;
        f.write_i64::<LE>(t_hi)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::C_HALF;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva_wf_pls_{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&d).unwrap();
        d.join("t.pls")
    }

    fn sample() -> Waveforms {
        let mut w = Waveforms::default();
        let d = crate::transform::normalize(&[0.2, 0.1, -1.0]);
        // Pulse 0: outgoing + two returning segments; pulse 1: one returning, no origin.
        let rows = [(0i64, 1u8, -2.0, 8u32), (0, 2, 600.0, 30), (0, 2, 700.0, 12), (1, 2, 5.0, 20)];
        for (k, &(p, _kind, off, n)) in rows.iter().enumerate() {
            w.pulse.push(p);
            w.gps_time.push(5000.25 + p as f64 * 1e-5);
            let known = p == 0;
            w.origin.push(if known { [100.0, 200.0, 900.0] } else { [f64::NAN; 3] });
            w.anchor.push(if known { [100.0, 200.0, 900.0] } else { [150.0, 210.0, 50.0] });
            w.direction.push(d);
            w.offset.push(off);
            w.interval.push(1.0);
            w.metres_per_ns.push(C_HALF);
            w.sample_start.push(w.samples.len());
            w.sample_count.push(n);
            w.samples.extend((0..n).map(|i| ((i * 13 + k as u32) % 200) as f32));
        }
        w.attrs.insert("kind".into(), Attr::U8(rows.iter().map(|r| r.1).collect()));
        w
    }

    #[test]
    fn round_trip() {
        let w = sample();
        let p = tmp("rt");
        write_waveforms(&w, &p, &PlsWriteOptions::default()).unwrap();
        let info = read_info(&p).unwrap();
        assert_eq!(info.n_pulses, 2);
        assert_eq!(info.descriptors.len(), 2);
        let (r, next) = read_waveforms(&p, &PlsReadOptions { kinds: Kinds::All, ..Default::default() }).unwrap();
        assert_eq!(next, 2);
        assert_eq!(r.len(), 4);
        assert_eq!(r.samples, w.samples);
        assert_eq!(r.pulse, vec![0, 0, 0, 1]);
        assert_eq!(r.attrs["kind"].to_f64(), vec![1.0, 2.0, 2.0, 2.0]);
        for (a, b) in w.sample_positions().iter().zip(r.sample_positions()) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 2e-3, "{a:?} {b:?}");
            }
        }
        assert!((r.origin[0][2] - 900.0).abs() < 1e-3);
        assert!(r.origin[3][0].is_nan());
        assert!((r.gps_time[3] - w.gps_time[3]).abs() < 1e-8);
        let (ret, _) = read_waveforms(&p, &PlsReadOptions::default()).unwrap();
        assert_eq!(ret.len(), 3);
        let (one, next) = read_waveforms(&p, &PlsReadOptions { start: 1, count: Some(5), ..Default::default() }).unwrap();
        assert_eq!((one.len(), next), (1, 2));
    }

    #[test]
    fn rejects_samples_out_of_range() {
        let mut w = sample();
        w.samples[0] = -3.0;
        assert!(write_waveforms(&w, tmp("bad"), &PlsWriteOptions::default()).is_err());
    }
}
